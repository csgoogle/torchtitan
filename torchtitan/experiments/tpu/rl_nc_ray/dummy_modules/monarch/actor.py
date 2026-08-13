import os

class Actor:
    pass
class Channel:
    pass
class Port:
    pass
class PortReceiver:
    pass
class HostMesh:
    pass
class ProcMesh:
    pass

def endpoint(func):
    return func
def this_host():
    pass

class RankProxy:
    @property
    def rank(self):
        return int(os.environ.get("RANK", "0"))

def current_rank():
    return RankProxy()
