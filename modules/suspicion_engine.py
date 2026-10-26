import time


class SuspicionEngine:

    def __init__(self):

        self.score = 0

        self.events = []

        self.last_event = {
            "looking_away": False,
            "phone": False,
            "multiple_faces": False
        }


    def update(
        self,
        looking_away,
        phone,
        multiple_faces
    ):

        current_events = {
            "looking_away": looking_away,
            "phone": phone,
            "multiple_faces": multiple_faces
        }


        for event_name, active in current_events.items():

            previous = self.last_event[event_name]


            if active and not previous:

                self.score += 1

                self.events.append({
                    "time": time.strftime("%H:%M:%S"),
                    "event": event_name,
                    "score": self.score
                })


        self.last_event = current_events


        if self.score == 0:

            status = "NORMAL"

        elif self.score <= 2:

            status = "SUSPICIOUS"

        else:

            status = "HIGHLY SUSPICIOUS"


        return self.score, status


    def get_events(self):

        return self.events