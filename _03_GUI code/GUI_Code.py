

#***************************************************************************************#

from tkinter import *
import os

from tkinter import * 
from tkinter import filedialog 
import shutil
from PIL import Image
import csv as cv
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import time

#***************************************************************************************#

path = "C:/Users/Administrator/Desktop/Image_Segmentation/_01_FCM/image"

#***************************************************************************************#

        
w1 = 0
data = 0
filename = ""
aaa = 0

def submit1():
    os.chdir(r"C:/Users/Administrator/Desktop/Image_Segmentation/_01_FCM")
    os.system('python FCM_GRAPHCUT.py')
    
    
    
def browseFiles():
    global w1
    global data
    
    
    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("Text files", 
                                                        "*.png*"), 
                                                       ("all files", 
                                                        "*.*"))) 
    print (filename)
    # Change label contents 
    #label_file_explorer.configure(text="File Opened: "+filename)
    shutil.move(filename, 'C:/Users/Administrator/Desktop/Image_Segmentation/_01_FCM/image')
    #images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']

                
                
#***************************************************************************************#
            
def exit_window():
    window.destroy()

#***************************************************************************************#
    
def Cacoon_Image():
    global window
    window = Toplevel(main_screen)
    window.title('SEGMENTATION OF MR IMAGE')  
    window.geometry("1000x500")  
    window.config(background = "white")


    global weight
    global wC

    weight = StringVar()

    Label(window,text = "SEGMENTATION OF MR IMAGE", width = 100, height = 4, font=("Calibri", 16) , bg = "blue" , fg="White"). pack()
    Label(window, text="").pack()
    


    Button(window,text = "Browse Files", font=("Calibri", 16) , command = browseFiles).pack()
    Label(window, text="").pack()
    
    Label(window, text="").pack()
    button_exit = Button(window,  text = "        Exit       ", font=("Calibri", 16) ,fg = "red",command = exit_window).pack()
    Label(window, text="").pack()

    Label(window, text="").pack()
    button_exit = Button(window,  text = "      Submit       ", font=("Calibri", 16) ,fg = "red",command = submit1).pack()
    Label(window, text="").pack()
    

    
#***************************************************************************************#  
    
def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("600x500")

    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()

    Label(register_screen, text="Please enter details below", bg="blue" ,width="300", height="2",font=("Calibri", 16) ).pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * " , font=("Calibri", 16))
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * " , font=("Calibri", 16))
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command = register_user , font=("Calibri", 16)).pack()


#***************************************************************************************# 

def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("600x500")
    Label(login_screen, text="Please enter details below to login" , bg="blue",width="300", height="2",font=("Calibri", 16)).pack()
    Label(login_screen, text="").pack()

    global username_verify
    global password_verify

    username_verify = StringVar()
    password_verify = StringVar()

    global username_login_entry
    global password_login_entry

    Label(login_screen, text="Username * " ,font=("Calibri", 16)).pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * " , font=("Calibri", 16)).pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command = login_verify ,font=("Calibri", 16)).pack()

#***************************************************************************************#

def register_user():

    username_info = username.get()
    password_info = password.get()

    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()

    username_entry.delete(0, END)
    password_entry.delete(0, END)

    Label(register_screen, text="Registration Success", fg="green", font=("calibri", 17)).pack()

#***************************************************************************************# 

def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)

    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            selection_page()

        else:
            password_not_recognised()

    else:
        user_not_found()

#***************************************************************************************#

def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()

#***************************************************************************************#
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()

#***************************************************************************************#

def delete_login_success():
    login_success_screen.destroy()

#***************************************************************************************#
    
def delete_password_not_recognised():
    password_not_recog_screen.destroy()

#***************************************************************************************#

def delete_user_not_found_screen():
    user_not_found_screen.destroy()


#***************************************************************************************#
    
def selection_page():
    global account_screen
    account_screen = Toplevel(main_screen)
    account_screen.geometry("600x500")
    account_screen.title("Select Option")
    Label(account_screen , text="Text Image", bg="blue", width="300", height="2", font=("Calibri", 16)).pack()
    Label(account_screen , text="").pack()
    Button(account_screen , text="Image Browse", height="2", width="30", command = Cacoon_Image , font=("Calibri", 16)).pack()

    main_screen.mainloop()

#***************************************************************************************#
    
def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("600x500")
    main_screen.title("Account Login")
    Label(text="SEGMENTATION OF MR IMAGE", bg="blue", width="300", height="2", font=("Calibri", 16)).pack()
    Label(text="").pack()
    Button(text="Login", height="2", width="30", command = login , font=("Calibri", 16)).pack()
    Label(text="").pack()
    Button(text="Register", height="2", width="30", command=register , font=("Calibri", 16)).pack()

    main_screen.mainloop()

#***************************************************************************************#
    
main_account_screen()

#***************************************************************************************#







