import datetime

class Timer:
    def __init__(self):
        self.t0      : datetime.datetime  = None
        self.t       : datetime.datetime  = None
        self.last_dt : datetime.timedelta = None
        self.counter : int                = 0

    # __init__

    @property
    def averaged_dt(self):
        if self.counter > 0:
            return self.total_elapsed_time / self.counter
        
        return None
    
    # property: averaged_dt

    @property
    def total_elapsed_time(self):
        return self.t - self.t0
    
    # property: total_elapsed_time

    def estimate_time_to_completion(self, num_left: int, averaged_dt: bool = False) -> datetime.timedelta:
        dt = self.last_dt
        if averaged_dt:
            dt = self.averaged_dt

        return dt * num_left
    
    # estimate_time_to_completion

    def reset(self):
        self.t0      = datetime.datetime.now()
        self.t       = self.t0
        self.last_dt = None
        self.counter = 0

    # reset

    def update(self):
        new_t         = datetime.datetime.now()
        self.last_dt  = new_t - self.t
        self.t        = new_t
        self.counter += 1

    # update

 # class: Timer