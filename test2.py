import matplotlib.pyplot as plt 
import numpy as np
print(plt.isinteractive())
#the function to turn on interactive mode 
plt.ion() 
print(plt.isinteractive())
#creating randomly generate collections/data 
random_array = np.arange(-4, 5) 
collection_1 = random_array ** 2
collection_2 = 10 / (random_array ** 2 + 1) 
figure, axes = plt.subplots() 
   
axes.plot(random_array, collection_1, 
          'rx', random_array, 
          collection_2, 'b+',  
          linestyle='solid') 
   
axes.fill_between(random_array,  
                  collection_1,  
                  collection_2, 
                  where=collection_2>collection_1,  
                  interpolate=True, 
                  color='green', alpha=0.3) 
   
lgnd = axes.legend(['collection-1', 
                    'collection-2'],  
                   loc='upper center',  
                   shadow=True) 
   
lgnd.get_frame().set_facecolor('#ffb19a') 
