import pickle as pk
import numpy as np
import spacy
import torch
import random
from sklearn.tree import _tree
import Searching_des as des
from reply import all_response_msg
from nn_model import NeuralNet

#******************************************* for Device ***********************************************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# remove warning 
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# ************************************************ import huggingface load & save as newsave_model *********************
# from transformers import pipeline
# PRETRAINED = "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed"
# ners = pipeline(task="ner",model=PRETRAINED, tokenizer=PRETRAINED)


# =====================================================  LOAD MODELS ==================================================
dtc , le  = pk.load(open('/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/chatbot_modelsave','rb'))
model,second_le  = pk.load(open('/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/chatbot_model','rb'))
ners = pk.load(open('/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/newsave_model','rb'))

# Load NN Model 
FILE = "/home/durgesh-dev/Desktop/Final Year Projects/chatbot/HealthCare_ChatBot/data.pth"
data = torch.load(FILE)

# -------------------------------------- initlising Global varible ----------------------
return_input_disease = []
if len(return_input_disease) > 1:
    return_input_disease.clear()
disease_first = []
symptoms_get = []
feature_names = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
nlp = spacy.load("en_core_web_sm")

#============================================== STEP 2 ===================================================
# pridected desiese Step 2
def input_output(present_disease):
    value_dicts = {
        '(vertigo) Paroymsal  Positional Vertigo': ['vomiting', 'headache', 'nausea', 'spinning_movements', 'loss_of_balance', 'unsteadiness'], 'AIDS': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'], 'Acne': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'], 'Alcoholic hepatitis': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1'], 'Allergy': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'], 'Arthritis': ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'painful_walking'], 'Bronchial Asthma': ['fatigue', 'cough', 'high_fever', 'breathlessness', 'family_history', 'mucoid_sputum'], 'Cervical spondylosis': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness', 'loss_of_balance'], 'Chicken pox': ['itching', 'skin_rash', 'fatigue', 'lethargy', 'high_fever', 'headache', 'loss_of_appetite', 'mild_fever', 'swelled_lymph_nodes', 'malaise', 'red_spots_over_body'], 'Chronic cholestasis': ['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'], 'Common Cold': ['continuous_sneezing', 'chills', 'fatigue', 'cough', 'high_fever', 'headache', 'swelled_lymph_nodes', 'malaise', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'loss_of_smell', 'muscle_pain'], 'Dengue': ['skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'malaise', 'muscle_pain', 'red_spots_over_body'], 'Diabetes ': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity', 'excessive_hunger', 'increased_appetite', 'polyuria'], 'Dimorphic hemmorhoids(piles)': ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'], 'Drug Reaction': ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_ urination'], 'Fungal infection': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic _patches'], 'GERD': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough', 'chest_pain'], 'Gastroenteritis': ['vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea'], 'Heart attack': ['vomiting', 'breathlessness', 'sweating', 'chest_pain'], 'Hepatitis B': ['itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine', 'loss_of_appetite', 'abdominal_pain', 'yellow_urine', 'yellowing_of_eyes', 'malaise', 'receiving_blood_transfusion', 'receiving_unsterile_injections'], 'Hepatitis C': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'yellowing_of_eyes', 'family_history'], 'Hepatitis D': ['joint_pain', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'], 'Hepatitis E': ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 'acute_liver_failure', 'coma', 'stomach_bleeding'], 'Hypertension ': ['headache', 'chest_pain', 'dizziness', 'loss_of_balance', 'lack_of_concentration'], 'Hyperthyroidism': ['fatigue', 'mood_swings', 'weight_loss', 'restlessness', 'sweating', 'diarrhoea', 'fast_heart_rate', 'excessive_hunger', 'muscle_weakness', 'irritability', 'abnormal_menstruation'], 'Hypoglycemia': ['vomiting', 'fatigue', 'anxiety', 'sweating', 'headache', 'nausea', 'blurred_and_distorted_vision', 'excessive_hunger', 'drying_and_tingling_lips', 'slurred_speech', 'irritability', 'palpitations'], 'Hypothyroidism': ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 'lethargy', 'dizziness', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'depression', 'irritability', 'abnormal_menstruation'], 'Impetigo': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'], 'Jaundice': ['itching', 'vomiting', 'fatigue', 'weight_loss', 'high_fever', 'yellowish_skin', 'dark_urine', 'abdominal_pain'], 'Malaria': ['chills', 'vomiting', 'high_fever', 'sweating', 'headache', 'nausea', 'diarrhoea', 'muscle_pain'], 'Migraine': ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision', 'excessive_hunger', 'stiff_neck', 'depression', 'irritability', 'visual_disturbances'], 'Osteoarthristis': ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain', 'swelling_joints', 'painful_walking'], 'Paralysis (brain hemorrhage)': ['vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'], 'Peptic ulcer diseae': ['vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases', 'internal_itching'], 'Pneumonia': ['chills', 'fatigue', 'cough', 'high_fever', 'breathlessness', 'sweating', 'malaise', 'phlegm', 'chest_pain', 'fast_heart_rate', 'rusty_sputum'], 'Psoriasis': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'], 'Tuberculosis': ['chills', 'vomiting', 'fatigue', 'weight_loss', 'cough', 'high_fever', 'breathlessness', 'sweating', 'loss_of_appetite', 'mild_fever', 'yellowing_of_eyes', 'swelled_lymph_nodes', 'malaise', 'phlegm', 'chest_pain', 'blood_in_sputum'], 'Typhoid': ['chills', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look_(typhos)', 'belly_pain'], 'Urinary tract infection': ['burning_micturition', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine'], 'Varicose veins': ['fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf'], 'hepatitis A': ['joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellowing_of_eyes', 'muscle_pain']
                    }
    symptoms_given = value_dicts[present_disease[0]]
    symptoms_get.clear()
    symptoms_get.append(symptoms_given)


# ================================================== NLTK, SPACY ==============================================
# step 0-A 
# Create Bag of words 
def bag_of_words(tokenize_sentence , all_words):
  bag = np.zeros(len(all_words),dtype=np.float32)
  for idx , w in enumerate(all_words):
    if w in tokenize_sentence:
      bag[idx] = 1.0
  return bag

# ------------------------------------- Nural Network Model load ------------------------------------
# nltk model data 
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

nn_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nn_model.load_state_dict(model_state)
nn_model.eval()

# Create Nltk model 
def nltk_output(msg):
    sentence = [token.lemma_ for token in nlp(msg)]
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = nn_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        return random.choice(all_response_msg[tag])
    return None


#************************************************ START TREE AND RECURESION TO FIND DIES ****************************************
# step 1-C
# get and add predicted dieases 
def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    disease_first.clear() 
    disease_first.append(disease)
    input_output(disease)

# step 1-B 
# recursion to get dieaseas input
def recurse(node, depth):
  tree = dtc
  feature_names = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
  disease_input = return_input_disease[0]
  tree_ = tree.tree_
  feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
  if tree_.feature[node] != _tree.TREE_UNDEFINED:
    name = feature_name[node]
    threshold = tree_.threshold[node]
    if name == disease_input:
      val = 1
    else:
      val = 0
    if  val <= threshold:
      recurse(tree_.children_left[node], depth + 1)
    else:
      recurse(tree_.children_right[node], depth + 1)
  else:
    present_disease = print_disease(tree_.value[node])
   
# step 1-A 
# Check pattern of input data and extract usefull information
def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    
    #input hugging face word
    try:
        des_word = []
        if len(des_word) > 1:
            des_word.clear()
        new_models = ners(f"i am suffering from {inp}" , aggregation_strategy="first")
        for result in new_models:
            if result['entity_group'] == "Disease" and int(float(result['score'])*100) >= 65 :
                    des_word.append(result['word'].replace(" ", ""))
        regexp = re.compile(des_word[0])
    except:
        des_word = []
        if len(des_word) > 1:
            des_word.clear()
        des_word.append(inp)
        try:       
            regexp = re.compile(des_word[0])
        except:
            des_word.append("joint pain")
            regexp = re.compile(des_word[0])
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if(len(pred_list)>0):
       # if match found then 
        return 1,pred_list,des_word[0]
    else:
        # if match not found or only one same type of item present
        return ptr,item,des_word[0]

# step 1
# using model check the passible diesiase can predict by tree and recursions after calling method
def tree_to_code(disease_input):
    conf,cnf_dis,desies=check_pattern(feature_names,disease_input)
    if conf==1:
        if len(cnf_dis) > 1:
            output = f"Are you suffering from which type of ' {desies} ' ? Please confirm that: \n"
            for num,it in enumerate(cnf_dis):
                output += f"\t --> {it} \n"
            output += f"\t Note: Please use underscore (  _  ) in place of spacing in the name of disease.\n"
            return [output]
        else :
            output = f"You are suffering from {desies}  \n"
            return_input_disease.clear()
            return_input_disease.append(disease_input) 
            recurse(0,1)
            return [output ,"get_des",symptoms_get]
    else:
        return ["Ohh!! There were no similar diseases discovered. Please enter a valid symptom."]

# ******************************************** Model Pridection Values ***************************************
#step 3
# Model Predictions send response
def get_pridected_value(symptoms_experiance):
  symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
  pred_des_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
  input_vector = np.zeros(len(symptoms_dict))
  for item in symptoms_experiance:
    input_vector[[symptoms_dict[item]]] = 1
  return pred_des_list[model.predict([input_vector])[0]]

# get diesease Description after model preductions
def get_diesese_practions(dieses):
    getprecaution = des.model.getPrecaution(dieses.capitalize())
    if disease_first[0][0] == dieses :
        getdescription = des.model.getDescription(dieses.capitalize())
        dieses =  dieses
    else:
        des_first = des.model.getDescription(disease_first[0][0].capitalize())
        des_sencond = des.model.getDescription(dieses.capitalize())
        getdescription = f"{des_sencond} \n\n {des_first}"
        dieses = f"{dieses} or {disease_first[0][0]}"
    
    output = f"You may have  {dieses} \n\n  {getdescription} \n  {getprecaution}\n"
    return output


# =============================================== RESPNSE SEND TO USER ========================================
# step 0
# send response to the user 
def getresponse(response_msg):
    response_msg = response_msg.lower()
    try:
        response_nn = nltk_output(response_msg)
    except:
        response_nn = None
    if response_nn is not None:
        return [response_nn]
    else:
        output_function = tree_to_code(response_msg)
        if len(output_function) > 1 :
            return output_function[0] ,output_function[2][0]
        elif len(output_function) == 1 :
            return output_function