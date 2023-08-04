from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import *
import os
import sys
import argparse
from PIL import Image
import cv2
import time
import database as db
from streamlit_option_menu import option_menu
globalkoshish =["trafficlight","chevron_markers","speedlimit","crosswalk","stop"]

# def checklist_app():
#     st.title("Streamlit Checklist App")
    
#     # Get user input
#     task = st.text_input("Add a new task:")
#     if st.button("Add"):
#         st.session_state.tasks.append(task)
    
#     # Display checklist
#     if 'tasks' not in st.session_state:
#         st.session_state.tasks = []
#     if len(st.session_state.tasks) > 0:
#         st.subheader("Task List:")
#         for i, task in enumerate(st.session_state.tasks):
#             st.write(f"{i+1}. {task}")
    
#     # Delete a task
#     if len(st.session_state.tasks) > 0:
#         st.subheader("Delete Task:")
#         task_to_delete = st.selectbox("Select a task to delete:", options=st.session_state.tasks)
#         if st.button("Delete"):
#             st.session_state.tasks.remove(task_to_delete)
# from hubconf import e
# from D:/semester 7/webApp/Yolo-v5-Streamlit-App-Pretrained-Model/yolov5/hubconf.py import results
# D:\semester 7\webApp\Yolo-v5-Streamlit-App-Pretrained-Model\yolov5\hubconf.py

#st.set_page_config(layout = "wide")
# st.set_page_config(page_title = "Yolo V5 Multiple Object Detection on Pretrained Model", page_icon="🤖")
# st.title("Main Page")
# st.sidebar.success("Select a page above.")
st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Capstone Project</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Automated Road Safety Auditing system through the intervention of AI</h2>", unsafe_allow_html=True)
# --- NAVIGATION MENU ---
selected = option_menu(
    menu_title=None,
    options=["Object Detection", "Retrieve Information"],
    icons=["pencil-fill", "bar-chart-fill"],  # https://icons.getbootstrap.com/
    orientation="horizontal",
)
# --- DATABASE INTERFACE ---
def get_all_periods():
    items = db.fetch_all_periods()
    periods = [item["key"] for item in items]
    return periods
if selected == "Object Detection":
    text_input = st.text_input(
            "Enter location and press Enter "
            # label_visibility=st.session_state.visibility,
            # disabled=st.session_state.disabled,
            # placeholder=st.session_state.placeholder,
        )
    if text_input:
        st.write("You entered: ", text_input)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 340px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 340px;
            margin-left: -340px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    #################### Title #####################################################
    #st.title('Yolo V5 Multiple Object Detection on Pretrained Model')
    #st.subheader('Multiple Object Detection on Pretrained Model')

    #st.markdown('---') # inserts underline
    #st.markdown("<hr/>", unsafe_allow_html=True) # inserts underline
    st.markdown('#') # inserts empty space

    #--------------------------------------------------------------------------------

    DEMO_VIDEO = os.path.join('data', 'videos', 'sampleVideo0.mp4')
    DEMO_PIC = os.path.join('data', 'images', 'bus.jpg')

    def get_subdirs(b='.'):
        '''
            Returns all sub-directories in a specific Path
        '''
        result = []
        for d in os.listdir(b):
            bd = os.path.join(b, d)
            if os.path.isdir(bd):
                result.append(bd)
        return result


    def get_detection_folder():
        '''
            Returns the latest folder in a runs\detect
        '''
        return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

    #---------------------------Main Function for Execution--------------------------

    def main():

        source = ("Detect From Image", "Detect From Video", "Detect From Live Feed")
        source_index = st.sidebar.selectbox("Select Activity", range(
            len(source)), format_func = lambda x: source[x])
        
        cocoClassesLst = ["trafficlight","chevron_markers","speedlimit","crosswalk","stop", "All"]
        
        classes_index = st.sidebar.multiselect("Select Classes", range(
            len(cocoClassesLst)), format_func = lambda x: cocoClassesLst[x])
        
        isAllinList = 80 in classes_index
        if isAllinList == True:
            classes_index = classes_index.clear()
            
        print("Selected Classes: ", classes_index)
        
        #################### Parameters to setup ########################################
        # MAX_BOXES_TO_DRAW = st.sidebar.number_input('Maximum Boxes To Draw', value = 5, min_value = 1, max_value = 5)
        deviceLst = ['cpu', '0', '1', '2', '3']
        DEVICES = st.sidebar.selectbox("Select Devices", deviceLst, index = 0)
        print("Devices: ", DEVICES)
        MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4)
        #################### /Parameters to setup ########################################
        
        weights = os.path.join("weights", "yolov5s.pt")

        if source_index == 0:
            
            uploaded_file = st.sidebar.file_uploader(
                "Upload Image", type = ['png', 'jpeg', 'jpg'])
            
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text = 'Resource Loading...'):
                    st.sidebar.text("Uploaded Pic")
                    st.sidebar.image(uploaded_file)
                    picture = Image.open(uploaded_file)
                    picture.save(os.path.join('data', 'images', uploaded_file.name))
                    data_source = os.path.join('data', 'images', uploaded_file.name)
            
            elif uploaded_file is None:
                is_valid = True
                st.sidebar.text("DEMO Pic")
                st.sidebar.image(DEMO_PIC)
                data_source = DEMO_PIC
            
            else:
                is_valid = False
        
        elif source_index == 1:
            
            uploaded_file = st.sidebar.file_uploader("Upload Video", type = ['mp4'])
            
            if uploaded_file is not None:
                is_valid = True
                with st.spinner(text = 'Resource Loading...'):
                    st.sidebar.text("Uploaded Video")
                    st.sidebar.video(uploaded_file)
                    with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    data_source = os.path.join("data", "videos", uploaded_file.name)
            
            elif uploaded_file is None:
                is_valid = True
                st.sidebar.text("DEMO Video")
                st.sidebar.video(DEMO_VIDEO)
                data_source = DEMO_VIDEO
            
            else:
                is_valid = False
        
        else:
            ######### Select and capture Camera #################
            
            selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "Use Other Camera"), index = 0)
            if selectedCam:
                if selectedCam == "Use Other Camera":
                    data_source = int(1)
                    is_valid = True
                else:
                    data_source = int(0)
                    is_valid = True
            else:
                is_valid = False
            
            st.sidebar.markdown("<strong>Press 'q' multiple times on camera window and 'Ctrl + C' on CMD to clear camera window/exit</strong>", unsafe_allow_html=True)
            
        if is_valid:
            print('valid')
            if st.button('Detect'):
                if classes_index:
                    with st.spinner(text = 'Inferencing, Please Wait.....'):
                        run(weights = weights, 
                            source = data_source,  
                            #source = 0,  #for webcam
                            conf_thres = MIN_SCORE_THRES,
                            #max_det = MAX_BOXES_TO_DRAW,
                            device = DEVICES,
                            save_txt = True,
                            save_conf = True,
                            classes = classes_index,
                            nosave = False, 
                            )
                            
                else:
                    with st.spinner(text = 'Inferencing, Please Wait.....'):
                        run(weights = weights, 
                            source = data_source,  
                            #source = 0,  #for webcam
                            conf_thres = MIN_SCORE_THRES,
                            #max_det = MAX_BOXES_TO_DRAW,
                            device = DEVICES,
                            save_txt = True,
                            save_conf = True,
                        nosave = False, 
                        )

                if source_index == 0:
                    with st.spinner(text = 'Preparing Images'):
                        for img in os.listdir(get_detection_folder()):
                            if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                                pathImg = os.path.join(get_detection_folder(), img)
                                st.image(pathImg)
                        
                        st.markdown("### Output")
                        st.write("Path of Saved Images: ", pathImg)    
                        st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels')) 
                        for i in range(len(globalkoshish)):
                            if dictkoshish[globalkoshish[i]]=="Yes":
                                checkbox_val = st.checkbox(globalkoshish[i],value=True)
                            else:
                                checkbox_val = st.checkbox(globalkoshish[i])
                        # for i in range(len(koshish)):
                        #     st.write("detected classes: ", koshish[i])
                        db.insert_period(text_input, koshish)
                        # st.header(f"Data Entry in ")
                        # with st.form("my_form"):
                        #     st.write("Inside the form")
                        #     checkbox_val = st.checkbox("Form checkbox")

                        #     # Every form must have a submit button.
                        #     submitted = st.form_submit_button("Submit")
                        #     if submitted:
                        #         st.write("slider", slider_val, "checkbox", checkbox_val)

                        #     st.write("Outside the form")
                        
                        # with st.form("entry_form"):
                        #     if "my_input" not in st.session_state:
                        #         st.session_state["my_input"] = ""
                        #     my_input = st.text_input("Input a text here", st.session_state["my_input"])
                        #     for i in range(len(koshish)):
                        #         checkbox_val = st.checkbox(koshish[i],value=True)
                        #     submit = st.form_submit_button("Submit")
                        #     if submit:
                        #         st.session_state["my_input"] = my_input
                        #         st.write("You have entered: ", my_input)
                        #         db.insert_period(my_input, koshish)
                        #         st.success("Data saved!")
                        # with st.form("entry_form", clear_on_submit=True):
                        #     col1, col2 = st.columns(2)
                        #     col1.selectbox("Select Month:", months, key="month")
                        #     col2.selectbox("Select Year:", years, key="year")    
                            # if agree:
                            #     st.write('Great!')     
                        # checklist_app()    
                        # st.balloons()
                        
                elif source_index == 1:
                    with st.spinner(text = 'Preparing Video'):
                        for vid in os.listdir(get_detection_folder()):
                            if vid.endswith(".mp4"):
                                #st.video(os.path.join(get_detection_folder(), vid))
                                #video_file = open(os.path.join(get_detection_folder(), vid), 'rb')
                                #video_bytes = video_file.read()
                                #st.video(video_bytes)
                                video_file = os.path.join(get_detection_folder(), vid)
                                
                    stframe = st.empty()
                    cap = cv2.VideoCapture(video_file)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    print("Width: ", width, "\n")
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print("Height: ", height, "\n")

                    while cap.isOpened():
                        ret, img = cap.read()
                        if ret:
                            stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                        else:
                            break
                    
                    cap.release()
                    st.markdown("### Output")
                    st.write("Path of Saved Video: ", video_file)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))
                    for i in range(len(globalkoshish)):
                            if dictkoshish[globalkoshish[i]]=="Yes":
                                checkbox_val = st.checkbox(globalkoshish[i],value=True)
                            else:
                                checkbox_val = st.checkbox(globalkoshish[i])    
                    # st.balloons()
                
                else:
                    with st.spinner(text = 'Preparing Video'):
                        for vid in os.listdir(get_detection_folder()):
                            if vid.endswith(".mp4"):
                                liveFeedvideoFile = os.path.join(get_detection_folder(), vid)
                        
                        st.markdown("### Output")
                        st.write("Path of Live Feed Saved Video: ", liveFeedvideoFile)    
                        st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels')) 
                        st.write("detected classes: ", pred)   
                        # st.balloons()
                    


    # --------------------MAIN FUNCTION CODE------------------------                                                                    
    if __name__ == "__main__":
        try:
            main()
            # print(results)
        except SystemExit:
            pass
    # ------------------------------------------------------------------
if selected =="Retrieve Information":
    st.header("Retrieving Information")
    with st.form("saved_periods"):
        period = st.selectbox("Select Period:", get_all_periods())
        submitted = st.form_submit_button("Get modules")
        if submitted:
            # Get data from database
            period_data = db.get_period(period)
            # st.write(period_data)
            # st.write(type(period_data))
            mdlist=period_data["modules"]
            for i in range(len(mdlist)):
                st.write("detected classes: ", mdlist[i])
            # for i in range len()
            #     st.write(period_data["modules"])    


