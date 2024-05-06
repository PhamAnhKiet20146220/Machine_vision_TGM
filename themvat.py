import cv2
import numpy as np
import tkinter as tk
from tkinter import Listbox, Button, messagebox

class ObjectManipulationApp:
    def __init__(self, master):
        self.master = master
        self.objects_to_add = ['anhTV.png', 'anhtulanh.png', 'anhbanghe.png', 'anhgiuong.png']
        self.background_image = cv2.imread('ktx1.jpg')
        self.objects_state = []
        self.selected_object_index = None
        self.start_x, self.start_y = -1, -1
        self.is_dragging = False
        self.result_window = None  # Keep track of the Result window

        self.create_widgets()

    def create_widgets(self):
        self.open_list_button = Button(self.master, text='Open Object List', command=self.open_object_list_window)
        self.open_list_button.pack()

        self.delete_button = Button(self.master, text='Delete Object', command=self.delete_object)
        self.delete_button.pack()

        self.open_result_button = Button(self.master, text='Open Result', command=self.open_result_window)
        self.open_result_button.pack()

        cv2.namedWindow('Result')
        cv2.setMouseCallback('Result', self.mouse_callback)

        self.update_background_image()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, obj_state in enumerate(self.objects_state):
                obj_x, obj_y = obj_state['position']
                if obj_x <= x <= obj_x + obj_state['size'][1] and obj_y <= y <= obj_y + obj_state['size'][0]:
                    self.selected_object_index = i
                    self.start_x, self.start_y = x - obj_x, y - obj_y
                    self.is_dragging = True
                    break

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            if self.selected_object_index is not None:
                obj_state = self.objects_state[self.selected_object_index]
                obj_height, obj_width = obj_state['size']

                new_x, new_y = x - self.start_x, y - self.start_y
                new_x = max(0, min(self.background_image.shape[1] - obj_width, new_x))
                new_y = max(0, min(self.background_image.shape[0] - obj_height, new_y))
                obj_state['position'] = (new_x, new_y)

                self.update_background_image()

    def open_object_list_window(self):
        object_list_window = tk.Toplevel(self.master)
        object_list_window.title('Object List')

        object_listbox = Listbox(object_list_window)
        for obj in self.objects_to_add:
            object_listbox.insert(tk.END, obj)

        def select_object():
            selected_object_path = self.objects_to_add[object_listbox.curselection()[0]]
            selected_object = cv2.imread(selected_object_path, cv2.IMREAD_UNCHANGED)

            obj_height, obj_width = selected_object.shape[:2]
            selected_object_resized = cv2.resize(selected_object, (obj_width, obj_height))

            self.objects_state.append({'image_resized': selected_object_resized, 'position': (0, 0),
                                       'size': (obj_height, obj_width)})

            self.update_background_image()
            object_list_window.destroy()

        select_button = Button(object_list_window, text='Select Object', command=select_object)
        select_button.pack()

        object_listbox.pack()

    def delete_object(self):
        if self.selected_object_index is not None:
            del self.objects_state[self.selected_object_index]
            self.selected_object_index = None
            self.update_background_image()
        else:
            messagebox.showinfo('Info', 'No object selected.')

    def update_background_image(self):
        self.background_image = cv2.imread('ktx1.jpg').copy()

        for obj_state in self.objects_state:
            obj_x, obj_y = obj_state['position']
            selected_object_resized = obj_state['image_resized']
            alpha_mask = selected_object_resized[:, :, 3] / 255.0

            min_row = max(0, obj_y)
            max_row = min(self.background_image.shape[0], obj_y + obj_state['size'][0])
            min_col = max(0, obj_x)
            max_col = min(self.background_image.shape[1], obj_x + obj_state['size'][1])

            M = np.array([
                [1, 0, obj_x - min_col],
                [0, 1, obj_y - min_row]
            ], dtype=np.float32)

            selected_object_resized = cv2.warpAffine(selected_object_resized, M, (max_col - min_col, max_row - min_row))

            for c in range(0, 3):
                self.background_image[min_row:max_row, min_col:max_col, c] = \
                    self.background_image[min_row:max_row, min_col:max_col, c] * (
                                1 - alpha_mask[:max_row - min_row, :max_col - min_col]) + \
                    selected_object_resized[:max_row - min_row, :max_col - min_col, c] * \
                    alpha_mask[:max_row - min_row, :max_col - min_col]

        cv2.imshow('Result', self.background_image)

    def open_result_window(self):
        if self.result_window is not None and not self.result_window.winfo_exists():
            # If the Result window is closed, set it to None
            self.result_window = None

        if self.result_window is None:
            
            self.update_background_image()
            
            cv2.namedWindow('Result')
            cv2.setMouseCallback('Result', self.mouse_callback)

            # Update the background image when reopening the Result window
            

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Main Window')

    app = ObjectManipulationApp(root)

    root.mainloop()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
