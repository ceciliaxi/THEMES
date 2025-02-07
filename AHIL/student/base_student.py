from .__head__ import *

import itertools
from agent  import BaseAgent, SerializableAgent
from buffer import ReplayBuffer

# REVISED HERE
# from .evaluation_utils import *

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



class BaseStudent(SerializableAgent):
    def __init__(self,
        env                  : gym.Env  ,
        trajs_path           : str      ,
        model_path           : str      ,
        run_seed             : int      ,
        batch_size           : int      ,
        buffer_size_in_trajs : int      ,
        teacher              : str, # REVISED HERE <-- (teacher: OAStableAgent)
        gym_env              : str,
        model_learner        : str,
    ):

        super(BaseStudent, self).__init__(
            env        = env       ,
            trajs_path = trajs_path,
            model_path = model_path,
        )

        self.run_seed             = run_seed
        self.batch_size           = batch_size
        self.buffer_size_in_trajs = buffer_size_in_trajs
        self.teacher              = teacher
        self.gym_env              = gym_env
        self.model_learner        = model_learner

        self._fill_buffer()


    # Match up the learned policy against state-actions pairs sampled from expert buffer
    def matchup(self) -> np.ndarray:
        samples = self.buffer.sample_all()
        state   = samples['state' ]
        action  = samples['action']

        # Select an action by EDMStudent
        action_hat = np.array([self.select_action(s) for s in state])
        match_samp =  np.equal(action, action_hat)
        return match_samp


    # -------------------------------------------------------------------------------------------------------
    # Two rollout functions:
    # Rollout function (Original)
    # NOTE: Only for Gym environment
    def rollout(self,
                max_step: int=200) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], float]:
        state = self.env.reset()
        traj = []
        retvrn = 0
        done = False
        count = 0

        while not done:
            student_state = state
            action, _, _ = self.select_action(student_state) # Select action by EDMStudent (TODO)
            reward, next_state, done = self.perform_action(action)
            if count >= max_step:
                break
            traj += [(state, action)]
            retvrn += reward
            state = next_state
            count += 1

        return traj, retvrn


    # Rollout function (For reformulated states)
    # NOTE: Only for Gym environment
    def rollout_reformfeat(self,
                           config: dict,
                           n: int=1,
                           reform_mode: str='original',
                           max_step: int=200) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], float]:
        state = self.env.reset()
        traj = []
        retvrn = 0
        done = False
        count = 0

        state_list = []
        while not done:
            # Normalize the states for MountainCar_v2 environment
            if 'MountainCar' in config['ENV']:
                # Load the pre-trained values from file to normalize he state
                path = config['BASE_PATH'] + 'contrib/baselines_zoo/trained_agents/ppo2/MountainCar-v0'
                with open("{}/{}.pkl".format(path, 'obs_rms'), 'rb') as file_handler:
                    obs_rms = pickle.load(file_handler)
                norm_state = self._normalize_obs(state, obs_rms)
            else:
                norm_state = state
            state_list.append(norm_state)

            # Transform the state
            student_state = state_converter(state_list, count, n, reform_mode)
            # Select action by EDMStudent
            action, _, _ = self.select_action(student_state)

            # Carry out the action
            reward, next_state, done = self.perform_action(action)
            if count >= max_step:
                break
            traj += [(state, action)]
            retvrn += reward
            state = next_state
            count += 1

        return traj, retvrn


    # # -------------------------------------------------------------------------------------------------------
    # # Two test functions
    # # Test the model in Gym environments by rolling out
    # def gym_test(self,
    #     config: dict,
    #     num_episodes : int,
    #     teacher_agent: OAStableAgent,
    #     n: int=1,
    #     reform_mode: str='original',
    #     save_traj: bool=False,
    # ) -> Tuple[float, float, float, int]:

    #     self.test_mode = True
    #     trajs = []
    #     matches = []
    #     returns = []
    #     for episode_index in range(num_episodes):
    #         if reform_mode == 'none':
    #             traj, retvrn = self.rollout()
    #         else:
    #             traj, retvrn = self.rollout_reformfeat(config, n, reform_mode)
    #         match = []
    #         for idx in range(len(traj)):
    #             state, action = traj[idx][0], traj[idx][1]
    #             match += [action == teacher_agent.select_action(state)]  # Select action by OAStableAgent
    #         trajs += [traj]
    #         matches += match
    #         returns += [retvrn]
    #     # Save rollout trajectories to file
    #     if self.env in self.gym_env and save_traj:
    #         np.save(self.trajs_path, {'trajs': trajs, 'returns': returns})

    #     return np.sum(matches) / len(matches), np.mean(returns), np.std(returns)


    # # Test the model in environments with other metrics
    # # Metrics: Accuracy, Recall, Precision, F1 score, AUC, APR, Jaccard score
    # def general_test(self,
    #     test_path : str,
    #     avg_patterns: str = 'weighted',
    # ) -> Tuple[float, float, float]:
    #     self.test_mode = True
    #     trajs = np.load(test_path, allow_pickle=True).item()['trajs']
    #     num_episodes = len(trajs)

    #     # REVISED HERE (02/14/2022)
    #     if self.model_learner  == 'MLP':
    #         X_test, y_test = GetFeatLabs(trajs)
    #         demo_action_list = list(y_test)
    #         # stu_action_list = list(self.getmlpmodel().predict(X_test))
    #         stu_action_pre_list = list(self.getmlpmodel().predict_proba(X_test))
    #         stu_action_list = list((self.getmlpmodel().predict_proba(X_test)[:, 1] > .35).astype('float'))




    #     elif self.model_learner  == 'EDM':
    #         demo_action_list, stu_action_list, stu_action_pre_list = [], [], []
    #         for episode_index in range(num_episodes):
    #             traj = trajs[episode_index]
    #             for idx in range(len(traj)):
    #                 state, action = traj[idx][0], traj[idx][1][0] # REVISED HERE (08/22/2021)
    #                 # Select action by EDMStudent (return the predicted action and the normalized qvalues)
    #                 state = np.array(state)
    #                 student_action, student_action_pred, _ = self.select_action(state)
    #                 demo_action_list.append(action)
    #                 stu_action_list.append(student_action.item())
    #                 stu_action_pre_list.append(student_action_pred)

    #     action_list = sorted(list(set(demo_action_list).union(set(stu_action_list))))

    #     # Calculate different metrics to evaluate the IL model
    #     _, metrics = ModelEvaluate(demo_action_list, stu_action_list, action_list,
    #                                avg_patterns=avg_patterns, verbo=True)

    #     AUC = AUCScore(demo_action_list, stu_action_list, stu_action_pre_list, avg_patterns=avg_patterns)
    #     APR = APScore(demo_action_list, stu_action_list, stu_action_pre_list, avg_patterns=avg_patterns)
    #     Jascore = JaccardScore(demo_action_list, stu_action_list, avg_patterns=avg_patterns)

    #     # TODO：Add the off-line evaluation methods:
    #     # Weighted doubly robust (WDR) & Off-policy classification (OPC)


    #     return metrics + [AUC] + [APR] + [Jascore], demo_action_list, stu_action_list, stu_action_pre_list


    # -------------------------------------------------------------------------------------------------------
    def serialize(self):
        raise NotImplementedError

    def deserialize(self):
        raise NotImplementedError


    # -------------------------------------------------------------------------------------------------------
    # Generate state-action pairs from demonstrations and put into the buffer
    def _fill_buffer(self, batch_norm=False):
        # REVISED HERE
        # Load the expert demonstrations (self.teacher <-- self.teacher.trajs_path)
        # trajs = np.load(self.teacher, allow_pickle=True)[()] \
        #             ['trajs'][self.run_seed:self.run_seed + self.buffer_size_in_trajs]

        teachers = np.load(self.teacher, allow_pickle=True)[()]
        # Slice the first #Trajs as Teachers
        trajs = teachers['trajs'][:self.buffer_size_in_trajs] ### REVISED HERE (09/07/2021)

        # Generate state-action pairs --> REVISED HERE (10/20/2021)
        if 'weights' not in teachers.keys():
            all_traj_num = len(teachers['trajs'])
            weights = [np.ones(len(teachers['trajs'][i])) for i in range(all_traj_num)]
        else:
            weights = teachers['weights'][:self.buffer_size_in_trajs]

        # Get all state-action pairs
        skip_step = 1 # Original: 20
        tmp_pairs = [[trajs[i][j] + (np.array([weights[i][j]]),) for j in range(len(trajs[i]))]
                 for i in range(len(trajs)) if i % skip_step == 0]
        pairs = list(itertools.chain.from_iterable(tmp_pairs))

        # Calculate the mean and std for all states
        all_states = list(itertools.chain.from_iterable([[trajs[i][j][0] for j in range(len(trajs[i]))]
                                                         for i in range(len(trajs))]))
        norm_states = np.mean(all_states, axis=0)
        std_states = np.std(all_states, axis=0)

        if len(pairs) < self.batch_size:
            self.batch_size = len(pairs)

        self.buffer = ReplayBuffer(
            state_dim  = self.env.observation_space.shape[0],
            total_size = len(pairs)                         ,
            batch_size = self.batch_size                    ,
        )

        for pair in pairs:
            if batch_norm:
                tmp_state = (pair[0] - norm_states) / std_states
            else:
                tmp_state = pair[0]

            self.buffer.store(
                state      = tmp_state, #pair[0],
                action     = pair[1],
                reward     = None   ,
                next_state = None   ,
                done       = None   ,
                weight     = pair[2],
            )

    # REVISED HERE:
    # Normalize observations using the VecNormalize's observations statistics.
    # Calling this method does not update statistics.
    # Reference: stable_baselines/common/vec_env/vec_normalize.py
    def _normalize_obs(self, obs: np.ndarray, obs_rms: np.ndarray, epsilon: float = 1e-8,
                       clip_obs: int = 10) -> np.ndarray:
        obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs)
        return obs