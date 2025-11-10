import events as e

def reward_from_events(self, events) -> int:
    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: pre defined events (events.py) that occured in game step

    return: reward based on events in (events.py)
    '''
    game_rewards = {
        e.COIN_COLLECTED: 100,        # Primary objective
        e.KILLED_OPPONENT: 500,        # High reward for eliminating opponents
        e.MOVED_RIGHT: -1,             # Small penalty to encourage efficiency
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -50,         # Increased from -10 to discourage invalid moves
        e.BOMB_DROPPED: -1,            # Small cost to prevent spam
        e.KILLED_SELF: -1000,          # CRITICAL: Heavily punish suicide (was 0!)
        e.GOT_KILLED: -700,            # Strong penalty for dying
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def rewards_from_own_events(self, events):
    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: pre defined events (events.py) that occured in game step

    return: reward based on own events
    '''
    reward_sum = 0
    reward_sum += crate_rewards(self, events)
    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum


def crate_rewards(self, events):
    '''
    Give the crate rewards imediately after placing the crates. This makes our agent place bombs more often (better)

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: pre defined events (events.py) that occured in game step

    return: reward based on how many crates will be destroyed by dropped bomb
    '''
    if e.BOMB_DROPPED in events:
        self.logger.info(f"reward for the {self.destroyed_crates} that are going to be destroyed -> +{self.destroyed_crates * 33}")
        return self.destroyed_crates * 33
    return 0