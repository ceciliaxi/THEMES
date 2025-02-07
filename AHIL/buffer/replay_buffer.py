from .base_buffer import *


class ReplayBuffer(BaseBuffer):
    def sample(self, sample_method='balanced') -> Dict[str, np.ndarray]:
        # REVISED HERE: TO SAMPLE BALANCED ACTIONS FROM THE BUFFER
        if sample_method == 'balanced':
            actions = np.unique(self.action_buf)
            action_per_num = int(self.batch_size / len(actions))

            # Check the number of each action
            action_num_dict = {action: len(np.where(self.action_buf == action)[0]) for action in actions}

            # Sort the actions by their occurred numbers
            argsort_list = np.argsort(list(action_num_dict.values()))

            indices = []
            for action_idx in argsort_list:
                tmp_action = list(action_num_dict.keys())[action_idx]
                # Get the number to be sampled for each action
                if action_num_dict[tmp_action] >= action_per_num:
                    tmp_size = action_per_num
                else:
                    tmp_size = action_num_dict[tmp_action]

                tmp_indices = np.random.choice(np.where(self.action_buf == tmp_action)[0], size=tmp_size, replace=False)
                indices.extend(list(tmp_indices))

            if len(indices) < self.batch_size:
                rem_indices = list(set(np.arange(len(self.action_buf))) - set(indices))
                tmp_indices = np.random.choice(rem_indices, size=self.batch_size - len(indices), replace=False)
                indices.extend(list(tmp_indices))

        elif sample_method == 'stratified':
            actions = np.unique(self.action_buf)
            action_per_num = int(self.batch_size / len(actions))

            # Check the number of each action
            action_num_dict = {action: len(np.where(self.action_buf == action)[0]) for action in actions}

            # Sort the actions by their occurred numbers
            argsort_list = np.argsort(list(action_num_dict.values()))

            indices = []
            for action_idx in argsort_list:
                tmp_action = list(action_num_dict.keys())[action_idx]
                tmp_action_num = int(self.batch_size * action_num_dict[tmp_action]/sum(action_num_dict.values()))

                # Get the number to be sampled for each action
                # if action_num_dict[tmp_action] >= tmp_action_num:
                tmp_size = tmp_action_num
                # else:
                #     tmp_size = action_num_dict[tmp_action]

                tmp_indices = np.random.choice(np.where(self.action_buf == tmp_action)[0], size=tmp_size, replace=False)
                indices.extend(list(tmp_indices))

            if len(indices) < self.batch_size:
                rem_indices = list(set(np.arange(len(self.action_buf))) - set(indices))
                tmp_indices = np.random.choice(rem_indices, size=self.batch_size - len(indices), replace=False)
                indices.extend(list(tmp_indices))


            if len(indices) < self.batch_size:
                add_indices = np.random.choice(set(np.where(self.action_buf == tmp_action)[0]) - set(tmp_indices),
                                               size=self.batch_size - len(indices), replace=False)
                indices.extend(add_indices)

        # Random sampling from all actions
        elif sample_method == 'random':
            indices = np.random.choice(self.total_size, size = self.batch_size, replace = False)

        return self._take_from(indices)
