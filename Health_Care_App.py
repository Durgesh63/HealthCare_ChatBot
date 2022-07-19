#Imports tkinter Library
from tkinter import *
import tkinter.messagebox
from tkinter.simpledialog import askstring

#Imports Library
import re
import webbrowser
from bot import getresponse,get_pridected_value , get_diesese_practions

# defined color for tkinter
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

#Main Tk class
class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self.outputs = []
        self.days = 0
        self._get_name()
        
    def run(self):
        self.window.mainloop()
    
    #--------------------------------------------- start Window asking questions ----------------------------------------------
    # Print answer to the tk window 
    def giveanswer(self,dises,ans):
        self.text_widget.tag_config('blue', foreground="#FDD20E")
        self.msg_entry.delete(0, END)
        sender = self.name_entry.get().split(" ")[0]
        msg1 = f"{sender} : "
        msg2 = f"{dises} -> {ans} \n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1.capitalize(), 'blue')
        self.text_widget.insert(END, msg2.capitalize())
        self.text_widget.configure(state=DISABLED)

    # asking function for how many days 
    def _suffering_days(self):
        no_of_days = askstring(f"Please only respond in a number of days.", f"Since how many days do you suffer? ")
        if no_of_days is None:
            self.msg_warning(f"Wrong Input ","Please respond only in number format. Do not cancel because it's important.")
            self._suffering_days()
        elif no_of_days.isnumeric() is True and int(no_of_days) > 0:
            self.giveanswer("Since how many days do you suffer? ",no_of_days)
            self.days += int(no_of_days)
        else:
            self.msg_warning(f"Wrong Input ","Please respond only in number format. Do not cancel because it's important.")
            self._suffering_days()


    # ask question to the user 
    def ask_box(self,desid):
        prompt = askstring(f"Please respond only with (Yes/No)", f"Are you suffering from a ' {desid} ' ? ")
        if prompt == None:
            self.msg_warning(f"{desid} Wrong Input ","Please respond only in (yes/no) format.Do not cancel because it's important.")
            self.ask_box(desid)
        elif prompt.lower() == "no" or prompt.lower() == "yes":
            self.giveanswer(desid,prompt)
            if prompt.lower() == "yes" :
                self.outputs.append(desid)
        else:
            self.msg_warning(f"{desid} Wrong Input ","Please give the answer only in (yes/no)")
            self.ask_box(desid)
    # -------------------------------------------End of ask question -----------------------------------------------


    # ====================================================  START OF TKINTER FIRST WINDOW =====================================================
    #First tkinter window open and asking the name of user
    def _get_name(self):
        self.window.title("Welcome to Health Care Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=500, height=500, bg=BG_COLOR)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, y=0, relheight=0.002)

        # entry label
        self.name = Label(self.window, bg="RED", fg=TEXT_COLOR,
                           text="Please Enter Your name", font=FONT_BOLD, pady=10)
        self.name.place(relwidth=1,y=57.3)

        # message entry box level
        name_label = Label(self.window, bg=BG_COLOR, height=80)
        name_label.place(relheight=0.09, relwidth=1,y=105.9)
        #msg entery box
        self.name_entry = Entry(name_label, bg=BG_GRAY, fg="BLACK", font=FONT_BOLD)
        self.name_entry.place(relheight=0.85,relwidth=0.74, x=65)
        self.name_entry.focus()
        self.name_entry.bind("<Return>", self.get_name_after_click)

        #save name button label
        name_box_label = Label(self.window, bg=BG_COLOR, height=80)
        name_box_label.place(relheight=0.09, relwidth=1,y=150.9)
        #button
        name_send_button = Button(name_box_label, text="Submit", fg=TEXT_COLOR, font=FONT_BOLD, width=20, bg="RED",
                             command=lambda: self.get_name_after_click(None))
        name_send_button.place(relheight=1,relwidth=0.24, x=190)

        #------------------------------- menu button ------------------------------------
        menu = Menu(self.window,bg=BG_COLOR,borderwidth=0,fg=TEXT_COLOR,font="bold")
        self.window.config(menu=menu, bd=5)

        # File menu 
        File = Menu(menu, tearoff=0,font="bold",activebackground="#FFFFFF")
        menu.add_cascade(label="File", menu=File,font="bold")
        File.add_command(label="Clear Chat",command=self.clear_chat,font="bold")
        File.add_command(label="Exit",command=None,font="bold")

        # About menu 
        about = Menu(menu, tearoff=0,font="bold",activebackground="#FFFFFF")
        menu.add_cascade(label="About", menu=about,font="bold")
        about.add_command(label="Develpoers", command=lambda: self.msg_showinfo(f"Bot Develpoers ",f"Project Mentor - Mr.Prashant Kumar Yadav \n\nGroup number - 01 \n\nGroup Members \n\n1. Durgesh Maurya (105501) \n2. Dimpla (185506) \n3. Dev Sharan Yadav (185532) \n"),font="bold")
        about.add_command(label="About Project", command=lambda: self.msg_showinfo(f"Bot V1.0.0 ",f" \tAbout Project \n\nThe primary goal of this project is to forecast the disease so that patients can get the desired output according to their primary symptoms.\n\n GitHub Link \n\nhttps://github.com/Durgesh63/HealthCare_ChatBot.git"),font="bold")

        # Quit menu 
        menu.add_command(label ='Quit!',font="bold", command=lambda: self.msg_msg_askcancle(f"Ok Quit "," Are You sure? "))

    # ====================================================  END OF TKINTER FIRST WINDOW =====================================================

    # Name Validations
    def get_name_after_click(self,name):
        if len(self.name_entry.get()) >= 2:
            self._setup_main_window()
            self.coming_msg()
        else:
            self.msg_warning("Message Regarding Name Error","Please enter at least two words of a name.")
            

    #================================================ START MAIN WINDOW =========================================================
    def _setup_main_window(self):
        self.window.title("Welcome to Health Care Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=1200, height=640, bg=BG_COLOR)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, y=0, relheight=0.012)
        
        #------------------- left side label -------------------------
        leftside_label = Label(self.window, bg=BG_COLOR, height=80,border=1)
        leftside_label.place(relheight=1, relwidth=0.2519,y=3 )

        # sidebox name lable
        self.name = Label(leftside_label, bg=BG_COLOR, fg=TEXT_COLOR,
                           text=self.name_entry.get().capitalize(), font=FONT_BOLD, pady=10)
        self.name.place(relheight=0.07, relwidth=1,y=1)

        # sidebox Help box
        self.help = Label(leftside_label, bg="red", fg=TEXT_COLOR,
                           text="Help Search (Similar diseases)", font=FONT_BOLD, pady=10)
        self.help.place(relheight=0.07, relwidth=1,y=48)

        # search box label
        search_label = Label(leftside_label, bg=TEXT_COLOR, height=80)
        search_label.place(relheight=0.07, relwidth=1,y=94)

        #Help message entry box 
        self.help_entry = Entry(search_label, bg=BG_GRAY, fg="BLACK", font=FONT_BOLD)
        self.help_entry.place(relheight=0.85,relwidth=0.74, y=2.4)
        self.help_entry.focus()
        self.help_entry.bind("<Return>", self._on_enter_help_search)

        # send button
        help_send_button = Button(search_label, text="Search", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_help_search(None))
        help_send_button.place(relheight=0.85,relwidth=0.24,y=2.4, x=224)

        #Searchbox
        self.search_box = Text(leftside_label, width=20, height=2, bg="#00003d", fg=TEXT_COLOR,
                                font=FONT, padx=8, pady=8)
        self.search_box.place(relheight=.779, relwidth=1,y=140)
        self.search_box.configure(cursor="arrow", state=DISABLED)
        # scroll bar for search_box
        scrollsearch = Scrollbar(self.search_box)
        scrollsearch.place(relheight=1, relx=0.97)
        scrollsearch.configure(command=self.search_box.yview)

        #------------------------------ righ sider lebel -------------------------------
        rightside_label = Label(self.window, bg=BG_COLOR, height=80,border=1)
        rightside_label.place(relheight=1, relwidth=0.75,y=3 ,x=300)

        # text widget
        self.text_widget = Text(rightside_label, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.893, relwidth=1,y=0 ,x=0)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.99)
        scrollbar.configure(command=self.text_widget.yview)

        
        #---------------------------------- bottom label ------------------------------------
        bottom_label = Label(rightside_label, bg="#17202A", height=80)
        bottom_label.place(relwidth=1,relheight=.1,x=1, y=563.7 )
        
        # message entry box #2C3E50
        self.msg_entry = Entry(bottom_label, bg=BG_GRAY, fg="Black", font=FONT)
        self.msg_entry.place(relheight=0.85,relwidth=0.82,x=5, y=4)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relheight=0.85, relwidth=0.15,x=745,y=4)

    #================================================ END START MAIN WINDOW =========================================================
    
    # get search help Diesies
    def _on_enter_help_search(self,event):
        dis_list = ['itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs', 'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
        inp = self.help_entry.get()
        pred_list=[]
        if len(inp) > 0:
            regexp = re.compile(inp)
            for item in dis_list:
                if regexp.search(item):
                    pred_list.append(item)
        self.help_entry.delete(0, END)
        if len(pred_list) > 0:
            msg1 = ""
            for i in range(len(pred_list)):
                msg1 += f"{i+1})  {pred_list[i]} \n"
        else:
            msg1 = "Ohh!! There were no similar diseases discovered."
        self.search_box.configure(state=NORMAL)
        self.search_box.delete("1.0",END)
        self.search_box.insert(END, msg1)
        self.search_box.configure(state=DISABLED)

    # bot First welcome Message 
    def coming_msg(self):
        good_name  = self.name_entry.get().split(" ")[0]
        msg = f"Hey! {good_name} , Are you not feeling well? Please tell me what symptoms here are some examples:\n --> fever\n --> cold\n --> cough\n --> headache\n --> stomach_pain\n --> abdominal_pain\n --> dehydration\n --> swelling\n --> acidity\n --> internal_itching\n --> sneezing\n --> vomiting\n --> anxiety , etc \n Note: Please use underscore (  _  ) in place of spacing in the name of disease.\n\n"
        # msg = f"Hey! {good_name} ,Are you not feeling well? Please tell me what symptoms here are some examples: \n fever , cold , cough , headache , stomach_pain , abdominal_pain , dehydration , swelling , acidity , \n itching , sneezing , vomiting , anxiety , etc.\nNote: Please use underscore( _ ) in place of spacing in the name of disease.\n\n"
        self._bot_insert(msg)

    #===================================== START RESPONSE MESSAGE =============================================
    # get the type msg from the entery of main msg box 
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg)

    #get and send response message     
    def _insert_message(self, msg):
        if not msg:
            return

        # delete the entery whtich is in entry box 
        self.msg_entry.delete(0, END)

        # sender msg or user msg 
        sender = self.name_entry.get().split(" ")[0]
        msg1 = f"{sender} : "
        msg2 = f"{msg}\n"

        #user insert msg
        self._user_instet_msg(msg1,msg2)

        # if we type any one from quit_msg then window close 
        quit_msg = ["quit","exit","bye","bye bye"]        
        if len([i for i in quit_msg if i == msg.lower()]) == 1:
            self.msg_msg_askcancle(f"Ok Quit "," Are You sure? ")
        
        # get msg form response from bot 
        else:
            chat_hear = getresponse(msg)
            if len(chat_hear) == 1 :
                msg4 = f"{chat_hear[0]} \n\n"
                self._bot_insert(msg4)
            elif len(chat_hear) > 1 :
                msg4 = f"{chat_hear[0]} \n \t Please give input on the diseases. \n\n"
                self._bot_insert(msg4)
                self.days *= 0
                self._suffering_days()
                self.outputs.clear()
                for i in chat_hear[1]:
                    self.ask_box(i)
                msg4 = f"You may also have diseases like \n"
                # if more than one output then condition true 
                if len(self.outputs) > 0 :
                    for i in range(len(self.outputs)):
                        msg4 += f"\t {i+1} ) : {self.outputs[i]}\n"
                    msg4 += f"\n"
                else:
                    msg4 += f"\t 1 ) : {chat_hear[0]}\n\n"
                self._bot_insert(msg4)

                # no of days 
                if self.days > 10:
                    self._bot_insert("Stop taking the medicine and reach out to the nearest hospital. \n")

                # get answer 
                final_dieses = get_pridected_value(self.outputs)
                try:
                    diesese_is =  get_diesese_practions(final_dieses)
                except:
                    diesese_is = f"Sorry no diese get {final_dieses} \n"
                self._bot_insert(diesese_is)

                # open webbrowser when number of days is grater than 10
                if self.days > 10:
                    ask_to_web = tkinter.messagebox.askokcancel("Permission to open Google Maps", "Do you want to launch Google Maps in your regular browser?\n ")
                    if ask_to_web is True:
                        webbrowser.open_new_tab('https://www.google.com/maps/search/hospital+near+me/')
                        self._bot_insert(" Open Google Maps on your default web browser \n")
            else:
                pass

    # ============================================= END RESPONSE MESSAGE ============================================================

    # clear chat box
    def clear_chat(self):
        self.text_widget.config(state=NORMAL)
        self.text_widget.delete(1.0, END)
        self.text_widget.delete(1.0, END)
        self.text_widget.config(state=DISABLED)
        self.coming_msg()
    # Insert msg from user
    def _user_instet_msg(self,usr,msg):
        self.text_widget.tag_config('blue', foreground="#FDD20E")
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, usr.capitalize(), 'blue')
        self.text_widget.insert(END, msg.capitalize())
        self.text_widget.configure(state=DISABLED)
    #insert msg from bot
    def _bot_insert(self,msg):
        self.text_widget.tag_config('red', foreground="#F93822")
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, "Bot : ","red")
        self.text_widget.insert(END, msg)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
    # ----------------------------------- Msg showing window --------------------------------    
    def msg_showinfo(self,title,msg):
        tkinter.messagebox.showinfo(title,msg)
    def msg_warning(self,title,msg):
        tkinter.messagebox.showwarning(title,msg)
    def msg_msg_askcancle(self,title,msg):
        msg_data = tkinter.messagebox.askokcancel(title, msg)
        if msg_data == True:
            self.window.destroy()
    # ----------------------------------- end Msg showing window --------------------------------        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()