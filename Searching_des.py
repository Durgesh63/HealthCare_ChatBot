import File_diesies_desc
import File_symptom_precaution 
class Searching:
    def __init__(self,diesis_desc,diesis_prec):
        self.diesis_desc_data = diesis_desc
        self.diesis_prec_data = diesis_prec

    # get Description of Desise 
    def getDescription(self,diesies):
        return self.diesis_desc_data[diesies]

    # get Precaution of Desise 
    def getPrecaution(self,diesies):
        return self.diesis_prec_data["Durg reaction"] + self.diesis_prec_data[diesies]
    
    def constants(self,numb):
        if int(numb) > 10:
            return "Consult Nearest Hospital and Stop Taking Drugs"
        else:
            return "Mild Desies Not to be Warry"

model = Searching(File_diesies_desc.diesis_desc,File_symptom_precaution.diesis_prec)

