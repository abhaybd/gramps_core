from gramps_core.utils import read_raw_log
from gramps_core.state import State

def add_replay_function(state: State):
    state.handlers['p'] = _replay_pose
    state.handlers['P'] = _replay_joints
    state.modes["replay"] = __replay
    state.modes["wait_for_replay"] = __wait_for_replay
    state.replay_recording = None

def _replay_pose(_, state: State):
    state.replay_type = "pose"
    replay(state)

def _replay_joints(_, state: State):
    state.replay_type = "joints"
    replay(state)

def replay(state: State):
    state.wait_for_replay_pos = list(state.current_position)
    with state.lock:
        state.mode = "wait_for_replay"
        state.replay_recording = None
    while state.replay_recording is None:
        recording_path = input("Recording to replay: ")
        try:
            recording = list(read_raw_log(recording_path))
            with state.lock:
                state.replay_idx = 0
                state.replay_recording = recording
        except:
            print("Couldn't open given recording path!")
    state.mode = "replay"
    print("Starting replay!")

def __wait_for_replay(state, _):
    return state.wait_for_replay_pos

def __replay(state: State, _):
    with state._mutex:
        if state.replay_recording is not None:
            s = state.replay_recording[state.replay_idx]
            if state.replay_type == "pose":
                target_choppose = s["target_choppose"]
                target_pos = state.arm.ik(target_choppose, state.current_position)
            elif state.replay_type == "joints":
                target_pos = s["target_position"]
            state.replay_idx += 1
            if state.replay_idx == len(state.replay_recording):
                print(f"Finished replay!")
                state.wait_for_replay_pos = target_pos
                state.mode = "wait_for_replay"
        else:
            target_pos = state.current_position
    return target_pos
