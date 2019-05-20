import numpy as np

R = np.matrix([[-1, -1, -1, -1, 0, -1],
                   [-1, -1, -1, 0, -1, 100],
                   [-1, -1, -1, 0, -1, -1],
                   [-1, 0, 0, -1, 0, -1],
                   [0, -1, -1, 0, -1, 100],
                   [-1, 0, -1, -1, 0, 100]])

Q = np.matrix(np.zeros([6, 6]))

gamma = 0.8
initial_state = 1
num_of_episode = 100

def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range,1))
    return next_action

def updateQ(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)

    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value


def main():
    available_act = available_actions(initial_state)
    action = sample_next_action(available_act)
    updateQ(initial_state, action, gamma)

    #Training
    for i in range(num_of_episode):
        current_state = np.random.randint(0,int(Q.shape[0]))
        available_act = available_actions(current_state)
        action = sample_next_action(available_act)
        updateQ(current_state, action, gamma)

    print("Trained Q Matrix")
    print(Q/np.max(Q)*100)

    #Testing
    starting_state = 2
    goal_state = 5 #Fix. cannot change. unless you change also the goal state in the training

    current_state = starting_state
    steps_sequence = [current_state]

    while current_state != goal_state:
        next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state,]))[1]
        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size=1))
        else:
            next_step_index = int(next_step_index)

        steps_sequence.append(next_step_index)
        current_state = next_step_index

    print("------------Testing------------")
    print("Starting state:", starting_state)
    print("Goal state:", goal_state)
    print("Optimal steps:")
    print(steps_sequence)

if __name__ == "__main__":
    main()