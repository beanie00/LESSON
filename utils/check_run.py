def include_mission(env_name):
    if "GoToDoor" in env_name or "GoToObject" in env_name or "PutNear" in env_name or "LockedRoom" in env_name or "Fetch" in env_name:
        return True
    return False
