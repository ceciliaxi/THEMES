from .base_student import *

from agent import CUDAAgent
from network import StudentNetwork


class EDMStudent(BaseStudent, CUDAAgent):
    def __init__(self,
                 env: gym.Env,
                 trajs_path: str,
                 model_path: str,
                 run_seed: int,
                 batch_size: int,
                 buffer_size_in_trajs: int,
                 teacher: str, # REVISED HERE <-- (teacher: OAStableAgent)
                 qvalue_function: StudentNetwork,
                 adam_alpha: float,
                 adam_betas: List[float],
                 sgld_buffer_size: int,
                 sgld_learn_rate: float,
                 sgld_noise_coef: float,
                 sgld_num_steps: int,
                 sgld_reinit_freq: float,
                 gym_env: str,
                 sample_method: str='balanced', # REVISED HERE: 'balanced'/'random' when sampling from buffer
                 model_learner: str='MLP', # Learning model: sklearn MLP/EDM
                 mlp_widths: int=128, # Hidden layer / layer list for MLP
                 ):

        super(EDMStudent, self).__init__(
            env=env,
            trajs_path=trajs_path,
            model_path=model_path,
            run_seed=run_seed,
            batch_size=batch_size,
            buffer_size_in_trajs=buffer_size_in_trajs,
            teacher=teacher,
            gym_env=gym_env,
            model_learner=model_learner,
        )

        self.qvalue_function = qvalue_function.to(self.device)
        self.adam_alpha = adam_alpha
        self.adam_betas = adam_betas

        self.optimizer = optim.Adam(qvalue_function.parameters(),
                                    lr=self.adam_alpha,
                                    betas=self.adam_betas,
                                    weight_decay=1e-3,
                                    )

        self.sgld_buffer = self._get_random_states(sgld_buffer_size)
        self.sgld_learn_rate = sgld_learn_rate
        self.sgld_noise_coef = sgld_noise_coef
        self.sgld_num_steps = sgld_num_steps
        self.sgld_reinit_freq = sgld_reinit_freq

        self.sample_method = sample_method
        self.model_learner = model_learner
        self.mlp_widths = mlp_widths


    # Select an action given the state based on predicted qvalues
    def select_action(self,
                      state: np.ndarray,
                      ) -> np.ndarray:

        # REVISED HERE (02/14/2022)
        if self.model_learner == 'MLP':
            qvalues = self.clf.predict_proba(state.reshape(1, -1))
            action = qvalues.argmax()
            # action = (self.clf.predict_proba(state)[:, 1] > 0.15).astype('float')
            qvalue_array = qvalues

        elif self.model_learner == 'EDM':
            qvalues = self.qvalue_function(torch.FloatTensor(state).to(self.device))
            action = qvalues.argmax()
            action = action.detach().cpu().numpy()
            qvalue_array = qvalues.detach().cpu().numpy()

        # REVISED HERE
        # Calculate the normalized qvalues (For prediction usage)
        norm_qvalue =(qvalue_array - min(qvalue_array)) / (max(qvalue_array) - min(qvalue_array))
        norm_qvalue = norm_qvalue / sum(norm_qvalue)

        return action, norm_qvalue, qvalue_array


    # Train the EDM model
    def train(self,
              num_updates: int,
              ):

        if self.model_learner == 'MLP':
            tr_teachers = np.load(self.teacher, allow_pickle=True).item()
            tr_trajs, test_returns = tr_teachers['trajs'], tr_teachers['returns']
            X_train, y_train = GetFeatLabs(tr_trajs)
            self.clf = MLPClassifier(random_state=1, ### TODO： Check the random number here
                                     max_iter=2000,
                                     hidden_layer_sizes=(self.mlp_widths)).fit(X_train, y_train)
            return self.clf.loss_curve_

        elif self.model_learner == 'EDM':
            # REVISED HERE
            # Check the convergence of two parts in loss function
            loss_list, loss_pi_list, loss_rho_list = [], [], []
            for _ in tqdm(range(num_updates)):
                # Sample a set of data from the buffer
                # Note: Default sample_method = 'balanced', i.e., each action has the same number
                samples = self.buffer.sample(self.sample_method)

                # Normalize the data in batch
                # tmp_min, tmp_max = np.mean(samples['state'], axis=0), np.std(samples['state'], axis=0)
                # tmp_state = (samples['state'] - tmp_min)/(tmp_max - tmp_min)
                # samples['state'] = tmp_state

                loss, loss_pi, loss_rho = self._compute_loss(samples)
                loss_list.append(loss)
                loss_pi_list.append(loss_pi)
                loss_rho_list.append(loss_rho)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.env.close()
            return loss_list, loss_pi_list, loss_rho_list


    # Compute the total loss
    def _compute_loss(self,
                      samples: Dict[str, np.ndarray],
                      ) -> torch.Tensor:
        # Calculate the cross-entropy(ce) loss
        loss_pi = self._compute_ce_loss(samples)

        # State <-- State-action pairs sampled from the expert demonstrations
        state_p = torch.FloatTensor(samples['state']).to(self.device)
        # States sampled from persistent contrastive divergence (PCD) buffer
        state_q = self._sample_via_sgld()

        # Calculate the logsumexp (REVISED HERE --> 10/20/21)
        weight = torch.FloatTensor(samples['weight']).to(self.device)
        logsumexp_f_p = (self.qvalue_function(state_p).logsumexp(1)*weight).mean()
        logsumexp_f_q = (self.qvalue_function(state_q).logsumexp(1)*weight).mean()

        # Calculate the "occupancy" loss
        loss_rho = logsumexp_f_q - logsumexp_f_p

        # Combine the entropy loss and occupancy loss
        # - occupancy loss goes exponentially when it becomes larger (balance the two losses)
        loss = loss_pi + 0*(loss_rho + loss_rho ** 2) ######## REVISED HERE
        return loss, loss_pi, loss_rho


    # Generate states by sampling from persistent contrastive divergence (PCD) buffer
    def _sample_via_sgld(self) -> torch.Tensor:
        samples, indices = self._initialize_sgld()
        # Take the samples as node in Graph
        x_t = torch.autograd.Variable(samples, requires_grad=True)

        for t in range(self.sgld_num_steps):
            # Calculate the gradient (second term in Eq.13)
            # qvalue_function uses CUDAAgent
            grad_logsumexp = torch.autograd.grad(
                self.qvalue_function(x_t).logsumexp(1).sum(),
                [x_t], retain_graph=True)[0]

            # - sgld_learn_rate: alpha; sgld_noise_coef: sigma
            # - randn_like: Returns a tensor with the same size as input filled with
            #               random numbers from a normal distribution with mean 0 and variance 1.
            grad_term = self.sgld_learn_rate * grad_logsumexp
            rand_term = self.sgld_noise_coef * torch.randn_like(x_t)
            x_t.data += grad_term + rand_term

        # Detach from the current graph
        samples = x_t.detach()
        # Move the tensor from GPU to CPU
        self.sgld_buffer[indices] = samples.cpu()
        return samples

    #
    def _initialize_sgld(self) -> Tuple[torch.Tensor, List[int]]:
        # Sample batch_size samples from sgld_buffer
        indices = torch.randint(0,
                                len(self.sgld_buffer),
                                (self.batch_size,),
                                )

        buffer_samples = self.sgld_buffer[indices]
        # Generate batch_size random samples
        random_samples = self._get_random_states(self.batch_size)

        # Random masking with with 0.95 probability for buffer_sample and 0.05 probability for random_sample
        mask = (torch.rand(self.batch_size) < self.sgld_reinit_freq).float()[:, None]
        samples = (1 - mask) * buffer_samples + mask * random_samples
        return samples.to(self.device), indices


    # Randomly sample a state
    def _get_random_states(self,
                           num_states: int,
                           ) -> torch.Tensor:
        state_dim = self.env.observation_space.shape[0]

        return torch.FloatTensor(num_states, state_dim).uniform_(-1, 1)


    # Calculate the cross-entropy(ce) loss
    def _compute_ce_loss(self,
                         samples: Dict[str, np.ndarray],
                         ) -> torch.Tensor:
        state = torch.FloatTensor(samples['state']).to(self.device)
        action = torch.LongTensor(samples['action']).to(self.device)
        weight = torch.FloatTensor(samples['weight']).to(self.device)

        # Get the qvalues
        qvalues = self.qvalue_function(state)

        # Calculate the cross-entropy loss
        ### REVISED HERE (10/20/2021)
        # orig_loss = nn.CrossEntropyLoss()(qvalues, action)

        tmp_loss = nn.CrossEntropyLoss(reduction='none')(qvalues, action) * weight
        # class_weights = torch.FloatTensor([1, 1])
        # tmp_loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)(qvalues, action) * weight

        loss = torch.mean(tmp_loss)
        return loss


    # Save/load model parameters in files
    def serialize(self):
        torch.save(self.qvalue_function.state_dict(), self.model_path)

    def deserialize(self):
        self.qvalue_function.load_state_dict(torch.load(self.model_path))


    # Return the MLP model
    def getmlpmodel(self):
        return self.clf