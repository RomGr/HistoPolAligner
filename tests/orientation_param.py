import tkinter
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image  
import numpy as np
import os
from datetime import datetime

from rembg import remove


def ask_for_parameters(combined_data, data, histology_fnames_link, center_of_mass):
    selector_objects = {}
    for folder, _ in combined_data.items():
        root = Tk()
        root.geometry("1430x1100")
        canvas = Canvas(root, width= 1430, height= 780, bg="white")
        canvas.pack(expand = NO)

        selector_objects[folder] = SelectorTkinter(root, canvas, combined_data, data, folder, histology_fnames_link, center_of_mass)
        
        root.mainloop()

    return get_parameters_dict(selector_objects)


def get_parameters_dict(selector_objects):
    parameters = {}
    for folder, selector_object in selector_objects.items():
        param = {}
        param['angle'] = int(selector_object.rotation.get())
        param['flip'] = selector_object.flip_txt
        param['shrink'] = int(selector_object.shrink.get())
        param['x_offest'] = int(selector_object.x.get())
        param['y_offest'] = int(selector_object.y.get())
        parameters[folder] = param
    return parameters

class SelectorTkinter:
    def __init__(self, root, canvas, combined_data, data, folder, histology_fnames_link, center_of_mass):
        self.img_size = combined_data[folder]['histology_cropped'].rotate(180).size
        self.root = root
        self.canvas = canvas
        self.combined_data = combined_data
        self.data = data
        self.imgs = combined_data[folder]
        self.folder = folder
        self.histology_fnames_link = histology_fnames_link
        self.center_of_mass = center_of_mass
        self.flip_txt = None
        
        self._show_imgs()
        self._create_rotation_scale()
        self._create_flip_tool()
        self._create_shrink_tool()
        self._create_x_tool()
        self._create_y_tool()
        self._create_button()
        
    def _show_imgs(self):
        # 1. polarimetry @ 550nm
        self.polarimetry550 = self.data[self.histology_fnames_link[self.folder].split('\\')[-1]]['images']['polarimetry']
        self.img_550, self.label_550 = display_image(self.canvas,self.polarimetry550,'./temp/imgs/pol550.png',[25,45],
                                                     [235, 25], "Polarimetry @ 550nm", self.img_size)
        
        # 2. Histology image
        self.histology_cropped = self.imgs['histology_cropped']
        self.img_histology, self.label_histology = display_image(self.canvas,self.histology_cropped,
                                           './temp/imgs/histology.png',[500,45],[715, 25],"Histology", self.img_size)
        
        # 3. polarimetry @ 650nm
        self.polarimetry650 = self.data[self.histology_fnames_link[self.folder].split('\\')[-1]]['images']['polarimetry_650']
        self.img_650, self.label_650 = display_image(self.canvas,self.polarimetry650,
                                     './temp/imgs/pol650.png',[975,45],[1190, 25],"Polarimetry @ 650nm",self.img_size)    

        # 4. overlay @ 550nm
        self.overlaid_550nm = Image.blend(self.histology_cropped, self.polarimetry550.convert('RGB'), 0.85)
        self.overlaid_550, self.label_overlaid_550 = display_image(self.canvas,self.overlaid_550nm,
                                          './temp/imgs/ovd550.png',[25,430],[235, 410],"Overlaid w/ histology",self.img_size)
        
        # 5. Histology GM/WM labels
        self.histology_GM_WM = self.imgs['labels_GM_WM_cropped']
        self.img_GM_WM, self.label_img_GM_WM = display_image(self.canvas,self.histology_GM_WM,
                                          './temp/imgs/histo_GM_WM.png',[500,430],[715, 410],"GM/WM mask",self.img_size)

        # 6. overlay @ 650nm
        self.overlaid_650nm = Image.blend(self.histology_GM_WM, self.polarimetry650.convert('RGB'), 0.85)
        self.overlaid_650, self.label_overlaid_650 = display_image(self.canvas,self.overlaid_650nm,
                                          './temp/imgs/ovdGMWM.png',[975,430],[1190, 410],"Overlaid w/ GM/WM mask",self.img_size)
        
        
    def _apply_changes(self):
        if self._check_flip():
            self.change_img()
                
    def _close_window(self):
        self.root.destroy()
                
    def _check_flip(self):
        try:
            assert self.flip.get() == 'h' or self.flip.get() == 'hv' or self.flip.get() == 'vh' or self.flip.get() == 'v' or self.flip.get() == ''
            self.flip_txt = self.flip.get()
            return True
        except:
            messagebox.showerror(title='Incorrect flip parameter', 
                        message='The parameter for flip is not is the lis of possible values, please check.')
            return False
        
    def _create_button(self):
        self.button_apply_changes = Button(self.root, text ="Apply changes", command = self._apply_changes, height= 2, width=20)
        self.button_apply_changes.place(x=400, y=1010)
        
        self.button_close = Button(self.root, text ="Save the parameters", command = self._close_window, height= 2, width=20)
        self.button_close.place(x=760, y=1010)
        
    def _create_rotation_scale(self):
        self.rotation = DoubleVar()
        self.rotation_scale = Scale(self.root, variable = self.rotation, from_ = -180, to = 180, 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white')   
        self.rotation_scale_text = Label(self.root, text = "Rotation", font=('Helvetica', 15, 'bold'))
        self.rotation_scale_text.place(x=160, y=850)
        self.rotation_scale.place(x=100, y=900)
        # self.button_rotation_scale = Button(self.root, text ="Change rotation", command = self._apply_changes) 
        # self.button_rotation_scale.place(x=150, y=950)
        
        
    def _create_flip_tool(self):
        self.flip = Entry(self.root, width = 25, bg = "grey", fg = "white")
        self.flip_text = Label(self.root, text = "Flip \n(h, v or hv)", font=('Helvetica', 15, 'bold'))
        self.flip_text.place(x=370, y=830)
        self.flip.place(x=350, y=921)
        # self.button_flipping = Button(self.root, text ="Change flipping", command = self._apply_changes) 
        # self.button_flipping.place(x=380, y=950)
            
    
    def _create_shrink_tool(self):
        self.shrink = DoubleVar()
        self.shrink_scale = Scale(self.root, variable = self.shrink, from_ = 0, to = 150, 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white')
        self.shrink_scale.set(100)
        self.shrink_scale_text = Label(self.root, text = "Shrinkage", font=('Helvetica', 15, 'bold'))
        self.shrink_scale_text.place(x=620, y=850)
        self.shrink_scale.place(x=560, y=900)
        # self.button_shrink_scale = Button(self.root, text ="Change shrinkage (%)", command = self._apply_changes) 
        # self.button_shrink_scale.place(x=560, y=950)
        
        
    def _create_x_tool(self):
        self.x = DoubleVar()
        self.x_scale = Scale(self.root, variable = self.x, from_ = -50, to = self.img_size[0], 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white')  
        self.x_scale.set(0)
        self.x_scale_text = Label(self.root, text = "x offset", font=('Helvetica', 15, 'bold'))
        self.x_scale_text.place(x=850, y=850)
        self.x_scale.place(x=780, y=900)
        # self.button_x_scale = Button(self.root, text ="Change x offset", command = self._apply_changes) 
        # self.button_x_scale.place(x=780, y=950)
        
        
    def _create_y_tool(self):
        self.y = DoubleVar()
        self.y_scale = Scale(self.root, variable = self.y, from_ = -50, to = self.img_size[1], 
                                    orient = HORIZONTAL, length = 200, activebackground = 'grey', bg = 'grey', fg = 'white')
        self.y_scale.set(0)
        self.y_scale_text = Label(self.root, text = "y offset", font=('Helvetica', 15, 'bold'))
        self.y_scale_text.place(x=1070, y=850)
        self.y_scale.place(x=1000, y=900)
        # self.button_y_scale = Button(self.root, text ="Change y offset", command = self._apply_changes) 
        # self.button_y_scale.place(x=1000, y=950)
        
    def change_img(self):
        self.label_histology.destroy()
        self.img_histology_PIL = apply_transformation(self.histology_cropped, self.center_of_mass[self.folder], self.rotation.get(), 
                                   self.flip_txt, self.shrink.get(), self.x.get(), self.y.get(), selection = True)
        self.img_histology, self.label_histology = display_image(self.canvas,self.img_histology_PIL,
                                           './temp/imgs/histology.png',[500,45],[715, 25],"Histology", self.img_size)
        
        self.label_img_GM_WM.destroy()
        self.histology_GM_WM_PIL = apply_transformation(self.histology_GM_WM, self.center_of_mass[self.folder], self.rotation.get(), 
                                   self.flip_txt, self.shrink.get(), self.x.get(), self.y.get(), selection = True)
        self.img_GM_WM, self.label_img_GM_WM = display_image(self.canvas,self.histology_GM_WM_PIL,
                                           './temp/imgs/histo_GM_WM.png',[500,430],[715, 410],"GM/WM mask",self.img_size)
        
        self.label_overlaid_550.destroy()
        self.overlaid_550nm = Image.blend(self.img_histology_PIL, self.polarimetry550.convert('RGB'), 0.85)
        self.overlaid_550, self.label_overlaid_550 = display_image(self.canvas,self.overlaid_550nm,
                                          './temp/imgs/ovd550.png',[25,430],[235, 410],"Overlaid w/ histology",self.img_size)
        
        self.label_overlaid_650.destroy()
        self.overlaid_650nm = Image.blend(self.histology_GM_WM_PIL, self.polarimetry650.convert('RGB'), 0.85)
        self.overlaid_650, self.label_overlaid_650 = display_image(self.canvas,self.overlaid_650nm,
                                          './temp/imgs/ovdGMWM.png',[975,430],[1190, 410],"Overlaid w/ GM/WM mask",self.img_size)
        

def get_img_disp(path, image):
    image.save(path)
    img = PhotoImage(file=path)
    return img

def display_image(canvas,img,path,img_coord,txt_coord,text,img_size):
    img = get_img_disp(path, img)
    label = Label(image=img, width=img_size[0]/1.2, height=int(img_size[1]/1.2))
    label.place(x=img_coord[0], y=img_coord[1])
    canvas.create_text(txt_coord[0], txt_coord[1], text=text, fill="black", font=('Helvetica 15 bold'))
    return img, label

def resize_image_after_rotation(img):
    img = np.asarray(img)
    shape = (388,516,3)
    img_resized = np.zeros(shape)

    for idx, x in enumerate(img):
        idx_dummy = idx - ((img.shape[0] - shape[0]) / 2)
        if idx_dummy > 0 and idx_dummy < img.shape[0]:
            for idy, y in enumerate(x):
                idy_dummy = idy - ((img.shape[1] - shape[1]) / 2)
                if idy_dummy > 0 and idy_dummy < img.shape[1]:
                    if sum(y) == 255 * 3:
                        pass
                    else:
                        try:
                            img_resized[int(idx_dummy), int(idy_dummy)] = img[idx, idy]
                        except:
                            pass
    return Image.fromarray(img_resized.astype(np.uint8))

def resize_image_after_shrinkage(img, shrink):
    if shrink > 100:
        return resize_image_after_rotation(img)
    else:
        img = np.asarray(img)
        shape = (388,516,3)
        img_resized = np.zeros(shape)
        difference_x, difference_y = int((shape[0] - img.shape[0])/ 2), int((shape[1] - img.shape[1])/2)
        for idx, x in enumerate(img):
            idx_dummy = idx + difference_x
            for idy, y in enumerate(x):
                idy_dummy = idy + difference_y
                img_resized[int(idx_dummy), int(idy_dummy)] = img[idx, idy]
        return Image.fromarray(img_resized.astype(np.uint8))
    
def move_by_offset(img, x_offset, y_offset):
    img = np.asarray(img)
    shape = (388,516,3)
    img_resized = np.zeros(shape)
    for idx, x in enumerate(img):
        idx_dummy = idx + y_offset
        for idy, y in enumerate(x):
            idy_dummy = idy + x_offset
            try:
                img_resized[int(idx_dummy), int(idy_dummy)] = img[idx, idy]
            except:
                pass
    return Image.fromarray(img_resized.astype(np.uint8))


def apply_transformation(image, center_mass, angle_rotation=0, flip='', shrink = 100, x_offset = 0, y_offset = 0, selection: bool = False):
    """
    apply_transformation is the master function to transform the H&E and labels image (rotate, flip, crop and resize)

    Parameters
    ----------
    image : Pillow image
        the image to be transformed
    x : int
        the center of the image in the x axis
    y : int
        the center of the image in the y axis
    angle : int
        the angle to use to rotate the image
    flip : boolean
        wether or not the image should be mirrored (default: True)
        
    Returns
    -------
    img_new : Pillow image
        the transformed image 
    """
    if selection:
        img_new = image
    else:
        img_new = get_cropped_image(image, center_mass)
    
    # 1. apply the rotation_angle
    black = (0,0,0)
    if angle_rotation:
        rotated = img_new.rotate(angle_rotation, expand=True, fillcolor = black)
    else:
        rotated = img_new.rotate(0, expand=True, fillcolor = black)
    
    # 2. apply the flipping
    if flip == 'vh' or flip == 'hv':
        img_new = rotated.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        img_new = img_new.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip == 'v':
        img_new = rotated.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
    elif flip == 'h':
        img_new = rotated.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        img_new = rotated
    img_new = resize_image_after_rotation(img_new)
    
    # 3. shrink the image
    new_shape = (int(img_new.size[0] * shrink / 100), int(img_new.size[1] * shrink / 100))
    img_new = img_new.resize(size = new_shape)
    img_new = resize_image_after_shrinkage(img_new, shrink)

    # 4. move by offset
    img_new = move_by_offset(img_new, x_offset, y_offset)
    
    return img_new


def get_cropped_image(image, center_mass, final_shape = (516 * 2, 388 *2 ,3)):
    
    x_shape = 516 * 2
    y_shape = 388 * 2
    
    dummy_img = np.zeros(final_shape)
    img_array = np.array(image)
    
    for idx, x in enumerate(img_array):
        if np.abs(idx - center_mass[0]) <= x_shape / 2:
            idx_dummy = int(idx - center_mass[0] + x_shape / 2)
            
            for idy, y in enumerate(x):
                if np.abs(idy - center_mass[1]) <= y_shape / 2:
                    idy_dummy = int(idy - center_mass[1] + y_shape / 2)
                    try:
                        dummy_img[idx_dummy, idy_dummy] = img_array[idx, idy]
                    except:
                        pass
        else:
            pass
        
    cropped_img = np.zeros(dummy_img.shape)
    for idx, x in enumerate(dummy_img):
        for idy, y in enumerate(x):
            if sum(y) == 0:
                cropped_img[idx, idy] = [255,255,255]
            else:
                cropped_img[idx, idy] = dummy_img[idx, idy]
            
    image = Image.fromarray(cropped_img.astype(np.uint8)).rotate(-90, expand=True).resize((516, 388), Image.Resampling.NEAREST)
    return image


def process_image_pathology(image, image_labels, path, center_mass, angle, image_labels_GM_WM = None, flip = '', 
                            shrink = 100, x_offset = 0, y_offset = 0, selection: bool = False): 
    """
    process_image_pathology is the master function used to transform, resize and crop the polarimetry image as well as
    the labels image

    Parameters
    ----------
    image : 
    image
        the image to be transformed
    image_labels : Pillow image
        the labels image to be transformed
    image_labels_GM_WM : Pillow image
        the GM/WM labels image to be transformed
    path : str
        the path to the histology folder for the measurements considered
    center_mass : int
        the center of the image in the x and y axis
    color_code : dict
        the color code for the labels for tumor cell content
    color_code_GM_WM : dict
        the color code for the labels for the GM/WM
    angle : int
        the angle to use to rotate the image
    flip : str
        wether or not the image should be mirrored - 'hv' or 'vh' for vertical and horizonal, 'h' for horizontal, 
        'v' for vertical and '' for none of the two (default: '')

    Returns
    -------
    images : list of Pillow image
        the transformed H&E and labels images
    images_upscaled : list of Pillow image
        the transformed H&E and labels images, but upscaled
    """
    # check if the mask for the tissue to analyze exists
    path_mask = os.path.join(path, 'to_rm_ng.tif')
    file_exists = os.path.exists(path_mask)
    
    #  if yes, use it to remove the undesired pixels
    if file_exists:
        mask = np.array(Image.open(path_mask))
        img = np.array(image)
        for idx, row in enumerate(mask):
            for idy, pix in enumerate(row):
                if pix != 0:
                    img[idx, idy] = 0
        image = Image.fromarray(img).convert('RGB')

    # apply the transformation (i.e. rotation + flip) to the polarimetric image and the labels images
    image = apply_transformation(image,center_mass,angle,flip=flip,shrink=shrink,x_offset=x_offset,y_offset=y_offset,selection=selection)
    image_labels = apply_transformation(image_labels,center_mass,angle,flip=flip,shrink=shrink,x_offset=x_offset,y_offset=y_offset,selection=selection)
    try:
        image_labels_GM_WM = apply_transformation(image_labels_GM_WM,center_mass,angle,flip=flip,shrink=shrink,x_offset=x_offset,y_offset=y_offset,selection=selection)
    except:
        pass
 
    # replace the background with white color
    im = Image.new('RGB', image_labels.size, (255, 255, 255))
    im.paste(image_labels, None)
    image_labels = im
    try:
        im = Image.new('RGB', image_labels_GM_WM.size, (255, 255, 255))
        im.paste(image_labels_GM_WM, None)
        image_labels_GM_WM = im
    except:
        pass
    
    img_registration = remove_background_histology(image, image_labels)
                
    # resize the images to the shape (516,388) or if rgb (516,388,3) and create upscaled version of the images when necessary
    img_registration_rgb = Image.fromarray(img_registration)
    img_registration_rgb = img_registration_rgb.resize((516, 388), Image.Resampling.NEAREST)
    
    img_registration = Image.fromarray(img_registration).convert('L')
    img_registration = img_registration.resize((516, 388), Image.Resampling.NEAREST)
    img_labels = image_labels.resize((516, 388), Image.Resampling.NEAREST)
    try:
        image_labels_GM_WM = image_labels_GM_WM.resize((516, 388), Image.Resampling.NEAREST)
    except:
        pass
    
    try:
        images = [img_registration, img_registration_rgb, img_labels, image_labels_GM_WM]
    except:
        images = [img_registration, img_registration_rgb, img_labels]
    
    return images


def remove_background_histology(image, image_labels):
    # create the mask to mask the background for the histology image...
    mask = np.array(image_labels).sum(axis = 2) == 0
    
    # ... and apply the mask to the histology image
    img_registration = np.array(image)
    for idx, x in enumerate(np.array(image)):
        for idy, y in enumerate(x):
            if mask[idx, idy] == 1:
                img_registration[idx, idy] = [0, 0, 0]
    
    return img_registration