from tkinter import *
from tkinter import messagebox
from PIL import Image  
import os
import json
from rembg import remove
import shutil

from histopolalign.helpers import load_position_names, compare_two_lists
from histopolalign.prepare_images import FolderAlignHistology, apply_transformation


def ask_for_parameters(measurement: FolderAlignHistology, force_recompute: bool = False):
    """
    ask_for_parameters is the master function allowing to get the parameters to apply for the processing of the pathology image

    Parameters
    ----------
    measurement : FolderAlignHistology
        the measurement object containing the information about the pathology image
    force_recompute : bool
        whether to force the re-ask the parameters (default is False)
        
    Returns
    -------
    """
    try:
        # try to load the parameters from the json file
        f = open(measurement.positions_path)
        data = json.load(f)
        f.close()
        
        # check if all the fields are present
        assert compare_two_lists(load_position_names(), list(data.keys()))
        assert force_recompute is False
        
        # if yes, then set the parameters in the measurement object
        measurement._set_the_parameters(data)
        
    except:
        # if any of the previous steps failed, then ask for the parameters
        root = Tk()
        root.geometry("1680x1300")
        canvas = Canvas(root, width= 1630, height= 1300, bg="white")
        canvas.pack(expand = NO)
        
        # create the selector object
        selector_object = SelectorTkinter(root, canvas, measurement)
        root.mainloop()
        
        # get the parameters from the selector object
        measurement._set_the_parameters(selector_object)
        
        # and save the parameters in the json file
        selector_object._save_parameters(measurement)
        return selector_object


class SelectorTkinter:
    """
    creates the object that will be used to select the parameters for the processing of the pathology image
    """
    def __init__(self, root: Tk, canvas: Canvas, measurement: FolderAlignHistology):
        """
        initialize the object with some of the images paths

        Parameters
        ----------
        root : Tk
            the root object of the tkinter window
        canvas : Canvas
            the canvas object of the tkinter window
        measurement : FolderAlignHistology
            the measurement object containing the information about the pathology image
        """
        try:
            shutil.rmtree('./temp')
        except FileNotFoundError:
            pass
        try:
            os.mkdir('./temp')
            os.mkdir('./temp/imgs')
        except FileExistsError:
            pass
        
        self.root = root
        self.canvas = canvas
        
        # set different paramters (img_size, center_of_mass, polarimetry_550, polarimetry_650, histology_cropped, labels_GM_WM_cropped)
        self.img_size = measurement.histology_cropped.size
        self.center_of_mass = measurement.center_of_mass
        self.polarimetry_550 = measurement.img_polarimetry_gs
        self.polarimetry_650 = measurement.img_polarimetry_gs_650
        self.histology_cropped = measurement.histology_cropped
        self.labels_GM_WM_cropped = measurement.labels_GM_WM_cropped
        self.flip_txt = None
        
        # create the tkinter variables, show the images and create the tools and buttons
        self._show_imgs()
        self._create_rotation_scale()
        self._create_flip_tool()
        self._create_shrink_tool()
        self._create_x_tool()
        self._create_y_tool()
        self._create_button()
        
    def _show_imgs(self):
        """
        _show_imgs is the master function that will show the images in the tkinter window by calling the _display_image function
        
        Parameters
        ----------
            
        Returns
        -------
        """
        # 1. show the polarimetry @ 550nm
        self.img_550, self.label_550 = self._display_image(self.polarimetry_550, './temp/imgs/pol550.png', [45,45], [255, 25], "Polarimetry @ 550nm")
        
        # 2. Histology image
        self.img_histology, self.label_histology = self._display_image(self.histology_cropped, './temp/imgs/histology.png', [580,45], [795, 25], "Histology")
        
        # 3. polarimetry @ 650nm
        self.img_650, self.label_650 = self._display_image(self.polarimetry_650, './temp/imgs/pol650.png', [1115,45], [1330, 25], "Polarimetry @ 650nm")

        # 4. overlay @ 550nm
        self.overlaid_550nm = Image.blend(self.histology_cropped, self.polarimetry_550.convert('RGB'), 0.85)
        self.overlaid_550, self.label_overlaid_550 = self._display_image(self.overlaid_550nm, './temp/imgs/ovd550.png', [45,500], [255, 480], "Overlaid w/ histology")
        
        # 5. Histology GM/WM labels
        self.img_GM_WM, self.label_img_GM_WM = self._display_image(self.labels_GM_WM_cropped, './temp/imgs/histo_GM_WM.png',[580,500], [795, 480], "GM/WM mask")
        
        # 6. overlay @ 650nm
        self.overlaid_650nm = Image.blend(self.labels_GM_WM_cropped, self.polarimetry_650.convert('RGB'), 0.85)
        self.overlaid_650, self.label_overlaid_650 = self._display_image(self.overlaid_650nm, './temp/imgs/ovdGMWM.png', [1115,500], [1330, 480], "Overlaid w/ GM/WM mask")
        
    
    def _display_image(self, img, path, img_coord, txt_coord, text):
        """
        _display_image is the function that will show the one image in the tkinter window
        
        Parameters
        ----------
        img : Image
            the image to be shown
        path : str
            the path of the image in the temp folder
        img_coord : list
            the coordinates of the image
        txt_coord : list
            the coordinates of the text
        text : str
            the text to be shown
            
        Returns
        -------
        img : Image
            the image to be shown
        label : Label
            the label object of the image
        """
        canvas = self.canvas
        img_size = self.img_size
        
        # create the PhotoImage object
        img = self._get_img_disp(path, img)
        
        # create the image and the label
        label = Label(image=img, width=img_size[0], height=int(img_size[1]))
        label.place(x=img_coord[0], y=img_coord[1])
        
        # add the text below the image (label)
        canvas.create_text(txt_coord[0], txt_coord[1], text=text, fill="black", font=('Helvetica 15 bold'))
        return img, label
        
    def _get_img_disp(self, path, image):
        """
        _get_img_disp is used to load the image to be displayed
        
        Parameters
        ----------
        path : str
            the path of the image in the temp folder
        image : Image
            the image to be shown
            
        Returns
        -------
        img : PhotoImage
            the PhotoImage object of the image to be shown
        """
        image.save(path)
        img = PhotoImage(file=path)
        return img

    def _create_rotation_scale(self):
        """
        _create_rotation_scale is used to create the rotation selector scale
        """
        self.rotation = DoubleVar()
        self.rotation_scale = Scale(self.root, variable = self.rotation, from_ = -180, to = 180, 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white') 
        self.rotation_scale.set(0)  
        self.rotation_scale_text = Label(self.root, text = "Rotation", font=('Helvetica', 15, 'bold'))
        self.rotation_scale_text.config(bg="white")
        self.rotation_scale_text.place(x=240, y=950)
        self.rotation_scale.place(x=180, y=1000)

    def _create_flip_tool(self):
        """
        _create_rotation_scale is used to create the flip selector tool
        """
        self.flip = Entry(self.root, width = 25, bg = "grey", fg = "white")
        self.flip_text = Label(self.root, text = "Flip (h, v or hv)", font=('Helvetica', 15, 'bold'))
        self.flip_text.config(bg="white")
        self.flip_text.place(x=482, y=950)
        self.flip.place(x=480, y=1023)

    def _create_shrink_tool(self):
        """
        _create_shrink_tool is used to create the shrinkage selector tool
        """
        self.shrink = DoubleVar()
        self.shrink_scale = Scale(self.root, variable = self.shrink, from_ = 0, to = 150, orient = HORIZONTAL, length = 200, 
                                  activebackground = 'grey', bg = 'grey', fg = 'white')
        self.shrink_scale.set(100)
        self.shrink_scale_text = Label(self.root, text = "Shrinkage", font=('Helvetica', 15, 'bold'))
        self.shrink_scale_text.config(bg="white")
        self.shrink_scale_text.place(x=790, y=950)
        self.shrink_scale.place(x=740, y=1000)

    def _create_x_tool(self):
        """
        _create_x_tool is used to create the tool to select the x offset
        """
        self.x = DoubleVar()
        self.x_scale = Scale(self.root, variable = self.x, from_ = -self.img_size[0], to = self.img_size[0], 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white')  
        self.x_scale.set(0)
        self.x_scale_text = Label(self.root, text = "x offset", font=('Helvetica', 15, 'bold'))
        self.x_scale_text.config(bg="white")
        self.x_scale_text.place(x=1090, y=950)
        self.x_scale.place(x=1030, y=1000)
        
    def _create_y_tool(self):
        """
        _create_y_tool is used to create the tool to select the y offset
        """
        self.y = DoubleVar()
        self.y_scale = Scale(self.root, variable = self.y, from_ = -self.img_size[1], to = self.img_size[1], 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white')
        self.y_scale.set(0)
        self.y_scale_text = Label(self.root, text = "y offset", font=('Helvetica', 15, 'bold'))
        self.y_scale_text.config(bg="white")
        self.y_scale_text.place(x=1390, y=950)
        self.y_scale.place(x=1330, y=1000)

    def _create_button(self):
        """
        _create_button creates the two buttons: 1. apply the selected changes 2. save the parameters
        """
        self.button_apply_changes = Button(self.root, text ="Apply changes", font='sans 16 bold', command = self._apply_changes, height= 2, width=20)
        self.button_apply_changes.place(x=280, y=1150)
        self.button_close = Button(self.root, text ="Save the parameters", font='sans 16 bold', command = self._close_window, height= 2, width=20)
        self.button_close.place(x=1100, y=1150)
        self.button_reset = Button(self.root, text ="Reset parameters", font='sans 16 bold', command = self._reset_parameters, height= 2, width=20)
        self.button_reset.place(x=690, y=1150)
        
    def _apply_changes(self):
        """
        _apply_changes is used to apply the selected changes to the image
        """
        if self._check_flip():
            self._change_img()
                
    def _check_flip(self):
        """
        _check_flip is used to check that the input for the flip parameter is correct
        """
        try:
            # check that the input is correct
            assert self.flip.get() == 'h' or self.flip.get() == 'hv' or self.flip.get() == 'vh' or self.flip.get() == 'v' or self.flip.get() == ''
            self.flip_txt = self.flip.get()
            return True
        except:
            # else print an error message
            messagebox.showerror(title='Incorrect flip parameter', 
                        message="The parameter for flip is not is the lis of possible values: ['hv', 'h', 'v', ''], please check.")
            return False
        
    def _change_img(self):
        """
        _change_img is used to apply the selected changes to the images
        """
        # destroy the previous images
        self.label_histology.destroy()
        self.label_img_GM_WM.destroy()
        
        # apply the changes to the histology and labels images
        [self.img_histology_PIL, self.histology_GM_WM_PIL] = apply_transformation([self.histology_cropped, self.labels_GM_WM_cropped],self.center_of_mass,
                                                        self.rotation.get(), self.flip_txt, self.shrink.get(), self.x.get(), self.y.get(), selection = True)

        # display the new images
        self.img_histology, self.label_histology = self._display_image(self.img_histology_PIL, './temp/imgs/histology.png', [580,45], [795, 25], "Histology")
        self.img_GM_WM, self.label_img_GM_WM = self._display_image(self.histology_GM_WM_PIL, './temp/imgs/histo_GM_WM.png',[580,500], [795, 480], "GM/WM mask")
        
        
        # destroy the previous overlaid images
        self.label_overlaid_550.destroy()
        self.label_overlaid_650.destroy()
        
        # creates the new overlaid images and display them
        self.overlaid_550nm = Image.blend(self.img_histology_PIL, self.polarimetry_550.convert('RGB'), 0.85)
        self.overlaid_550, self.label_overlaid_550 = self._display_image(self.overlaid_550nm, './temp/imgs/ovd550.png', [45,500], [255, 480], "Overlaid w/ histology")
        self.overlaid_650nm = Image.blend(self.histology_GM_WM_PIL, self.polarimetry_650.convert('RGB'), 0.85)
        self.overlaid_650, self.label_overlaid_650 = self._display_image(self.overlaid_650nm, './temp/imgs/ovdGMWM.png', [1115,500], [1330, 480], "Overlaid w/ GM/WM mask")

    
    def _close_window(self):
        """
        _close_window is used to destroy the tkinter window
        """
        self.root.destroy()
        
    def _reset_parameters(self):
        """
        _close_window is used to destroy the tkinter window
        """
        self.rotation_scale.set(0)  
        self.shrink_scale.set(100)
        self.x_scale.set(0)
        self.y_scale.set(0)
        self.flip.delete(0, END)
                
    def _save_parameters(self, measurement):
        """
        save_parameters is used to save the parameters in the json file

        Parameters
        ----------
        measurement : FolderAlignHistology
            the measurement object containing the information about the pathology image
            
        Returns
        -------
        """
        # create the dictionary
        positions = {}
        positions['x'] = self.center_of_mass[0]
        positions['y'] = self.center_of_mass[1]
        positions['angle'] = int(self.rotation.get())
        positions['rotation'] = self.flip_txt
        positions['shrink'] = int(self.shrink.get())
        positions['x_offest'] = int(self.x.get())
        positions['y_offest'] = int(self.y.get())
                
        # save the dictionary in the json file
        try:
            os.mkdir(measurement.positions_path.split('/')[0])
        except FileExistsError:
            pass
        with open(measurement.positions_path, 'w') as f:
            json.dump(positions, f)
        f.close()