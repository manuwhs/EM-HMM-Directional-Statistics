
import sys
import os

base_path = os.path.abspath('')
print ("Base path: %s"%base_path)

sys.path.append(base_path)

sys.path.append(base_path + "/libs")
sys.path.append(base_path + "/libs/graph")       # Graphical libs
sys.path.append(base_path + "/libs/graph/GUI/")       # Graphical libs
sys.path.append(base_path + "/libs/graph/specific/")       # Graphical libs

# FOR THE PAPER !!

sys.path.append(base_path + "/libs/EM")
sys.path.append(base_path + "/libs/EM/EM POO")
sys.path.append(base_path + "/libs/EM/utils")
sys.path.append(base_path + "/libs/Distributions")
sys.path.append(base_path + "/libs/Distributions/Watson")
sys.path.append(base_path + "/libs/Distributions/Gaussian")
sys.path.append(base_path + "/libs/Distributions/vonMisesFisher")

sys.path.append(base_path + "/libs/utils")

#imp_folders(os.path.abspath(''))
    
# Change code to only make it one ? main1, change folder name and database
