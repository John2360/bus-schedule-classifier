import cv2

class BusMovementTracker:
    movement_dict = {}
    valid_tracks = []
    frame_counter = 0

    def update_tracker(self, id, position, image):

        if id in self.movement_dict:
            direction_list = self.movement_dict[id]["directions"]
            direction_list.append(self.left_or_right(position, self.movement_dict[id]["position"]))

            self.movement_dict[id] = {"directions": direction_list, "position": (position[0], position[1]), "last_seen": self.frame_counter, "last_image": image}
        else:
            self.movement_dict[id] = {"directions": [], "position": (position[0], position[1]), "last_seen": self.frame_counter, "last_image": image}

    def clean_up(self):
        self.frame_counter += 1

        for id in list(self.movement_dict):
            if self.frame_counter-self.movement_dict[id]["last_seen"] >= 30:

                if len(self.movement_dict[id]["directions"]) >= 2:
                    final_direction = self.final_direction(self.movement_dict[id]["directions"])
                    if len(self.valid_tracks) > 0 and self.valid_tracks[-1]["direction"] == final_direction:
                        if self.frame_counter - self.valid_tracks[-1]["frame"] >= 20:
                            try:
                                cv2.imwrite("../detections/"+str(self.frame_counter)+"_"+final_direction+"_bus.jpg", self.movement_dict[id]["last_image"])
                            except:
                                print("Error: failed to write bus image.")
                                
                            del self.movement_dict[id]
                            self.valid_tracks.append({"direction": final_direction, "frame": self.frame_counter})
                            
                    else:
                        try:
                            cv2.imwrite("../detections/"+str(self.frame_counter)+"_"+final_direction+"_bus.jpg", self.movement_dict[id]["last_image"])
                        except:
                            print("Error: failed to write bus image.")

                        del self.movement_dict[id]
                        self.valid_tracks.append({"direction": final_direction, "frame": self.frame_counter})


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