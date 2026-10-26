import time


class HeadTurnTracker:

    def __init__(self, threshold=25, duration=1.5):

        self.threshold = threshold
        self.duration = duration
        self.start_time = None
        self.event_active = False

    def update(self, yaw):

        turned = abs(yaw) > self.threshold

        if turned:

            if self.start_time is None:
                self.start_time = time.time()

            elapsed = time.time() - self.start_time

            if elapsed >= self.duration:

                self.event_active = True

                return True

        else:

            self.start_time = None
            self.event_active = False

        return False


class GazeTracker:

    def __init__(self, duration=1.5):

        self.duration = duration
        self.start_time = None
        self.event_active = False
        self.current_direction = "CENTER"

    def update(self, gaze_direction):

        if gaze_direction == "CENTER":

            self.start_time = None
            self.event_active = False
            self.current_direction = "CENTER"

            return False

        if self.current_direction != gaze_direction:

            self.start_time = time.time()
            self.event_active = False
            self.current_direction = gaze_direction

        if self.start_time is None:

            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        if elapsed >= self.duration:

            self.event_active = True

            return True

        return False


class MultipleFaceTracker:

    def __init__(self, duration=1.5):

        self.duration = duration
        self.start_time = None
        self.event_active = False

    def update(self, face_count):

        multiple_faces = face_count > 1

        if not multiple_faces:

            self.start_time = None
            self.event_active = False

            return False

        if self.start_time is None:

            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        if elapsed >= self.duration:

            self.event_active = True

            return True

        return False


class PhoneTracker:

    def __init__(self, duration=1.5):

        self.duration = duration
        self.start_time = None
        self.event_active = False

    def update(self, phone_detected):

        if not phone_detected:

            self.start_time = None
            self.event_active = False

            return False

        if self.start_time is None:

            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        if elapsed >= self.duration:

            self.event_active = True

            return True

        return False