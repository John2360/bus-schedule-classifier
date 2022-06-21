import cv2
import os
import time
from datetime import date, datetime

from tinydb import TinyDB
db = TinyDB('./detections/database.json')

class BusMovementTracker:
    movement_dict = {}
    valid_tracks = []
    prior_frames = []
    
    frame_counter = 0
    time_now = datetime.now()

    def update_tracker(self, id, position, image):

        if id in self.movement_dict:
            direction_list = self.movement_dict[id]["directions"]
            direction_list.append(self.left_or_right(position, self.movement_dict[id]["position"]))

            self.movement_dict[id] = {"directions": direction_list, "position": (position[0], position[1]), "first_seen": self.movement_dict[id]["first_seen"], "last_seen": datetime.now(), "last_image": image}
        else:
            self.movement_dict[id] = {"directions": [], "position": (position[0], position[1]), "first_seen": datetime.now(), "last_seen": datetime.now(), "last_image": image}

    def clean_up(self, raw_image, detections_image):
        self.frame_counter += 1
        self.time_now = datetime.now()

        for id in list(self.movement_dict):
            if (self.time_now-self.movement_dict[id]["last_seen"]).total_seconds() >= 5:

                print(len(self.movement_dict[id]["directions"]))
                if len(self.movement_dict[id]["directions"]) >= 2:
                    final_direction = self.final_direction(self.movement_dict[id]["directions"])
                    print(final_direction)

                    # TODO: Investiage what this really does?
                    if len(self.valid_tracks) > 0 and self.valid_tracks[-1]["direction"] == final_direction:
                        if (self.time_now - self.valid_tracks[-1]["time"]).total_seconds() >= 5:
                            try:
                                cv2.imwrite("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/"+str(self.movement_dict[id]["first_seen"].timestamp())+".jpg", self.movement_dict[id]["last_image"])
                            except:
                                print("Error: failed to write bus image.")

                            db.insert({'time': str(self.time_now.timestamp()), 'direction': final_direction, 'folder_location': "detections/"+str(self.movement_dict[id]["first_seen"].timestamp())})
                            del self.movement_dict[id]
                            self.valid_tracks.append({"direction": final_direction, "time": self.time_now})
                            
                    else:
                        try:
                            cv2.imwrite("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/"+str(self.movement_dict[id]["first_seen"].timestamp())+".jpg", self.movement_dict[id]["last_image"])
                        except:
                            print("Error: failed to write bus image.")

                        db.insert({'time': str(self.time_now.timestamp()), 'direction': final_direction, 'folder_location': "detections/"+str(self.movement_dict[id]["first_seen"].timestamp())})
                        del self.movement_dict[id]
                        self.valid_tracks.append({"direction": final_direction, "time": self.time_now})

            
            else:
                if not os.path.exists("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())):
                    os.makedirs("detections/"+str(self.movement_dict[id]["first_seen"].timestamp()))

                                        
                if not os.path.exists("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/raw"):
                    os.makedirs("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/raw")

                    for frame in self.prior_frames:
                        cv2.imwrite("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/raw/"+str(frame["time"].timestamp())+".jpg", frame["raw"])

                if not os.path.exists("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/labeled"):
                    os.makedirs("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/labeled")

                    for frame in self.prior_frames:
                        cv2.imwrite("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/labeled/"+str(frame["time"].timestamp())+".jpg", frame["labeled"])
                
                cv2.imwrite("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/raw/"+str(self.time_now.timestamp())+".jpg", raw_image)
                cv2.imwrite("detections/"+str(self.movement_dict[id]["first_seen"].timestamp())+"/labeled/"+str(self.time_now.timestamp())+".jpg", detections_image)
        
        self.prior_frames.insert(0, {"time": self.time_now, "raw": raw_image, "labeled": detections_image})

        while (self.time_now - self.prior_frames[-1]["time"]).total_seconds() >= 5:
            self.prior_frames.pop()



    def left_or_right(self, position_new, position_old):
        move1 = position_new[0]-position_old[0]
        move2 = position_new[1]-position_old[1]

        if move1 >= 0 and move2 >= 0:
            return 1
        elif move1 <= 0 and move2 <= 0:
            return -1
        else:
            return 0
    
    def final_direction(self, directions):
        if sum(directions)/len(directions) > 0:
            return "right"
        elif sum(directions)/len(directions) < 0:
            return "left"
        else:
            return "stop"