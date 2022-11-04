import tkinter as tk 
from tkinter import *
from tkinter import ttk
from Main import DrawStabilityFractalOpencl,DrawNewtonsFractalOpencl  
# Top level window
frame = tk.Tk()
frame.title("TextBox Input")
frame.geometry('600x400')
# Function for getting Input
# from textbox and printing it 
# at label widget
class mainwindow:
    def StabilityFrac(self):
        options=self.AllOptions["Stability"].copy()
        for i in options.keys():
            options[i]=options[i].get()
        if options['Cycle Detection']:
            cycles=10
        else:
            cycles=0
         
        if options["Tolerance for cycle Detection"].lower() != "auto": 
            try:
                cycleacc=float(options["Tolerance for cycle Detection"])
            except ValueError:
                cycleacc=None
                print("invalid cycle Accuracy, defaulting to auto")
        else:cycleacc=None
            
        extent=options['Extent/Corners in form x1,x2,y1,y2'].split(",")
        extent=list(map(float,extent))
        
        DrawStabilityFractalOpencl(*extent,
                                   options["Seed Function"].lower(),
                                   maxdepth=int(options['Max Iteration/Depth']),
                                   cycles=cycles,
                                   ittcountcolouring =options['Colour from Iteration Count'],
                                   npoints=int(options["SideLength if square (pixels)"]),
                                   cycleacc=cycleacc)
        
    def NewtonsFrac(self):
        options=self.AllOptions["Newtons"]
        for i in options.keys():
            options[i]=options[i].get()
        extent=options['Extent/Corners in form x1,x2,y1,y2'].split(",")
        extent=list(map(float,extent))
        DrawNewtonsFractalOpencl(*extent,
                                   options["Seed Function"].lower(),
                                   maxdepth=int(options['Max Iteration/Depth']),
                                   npoints=int(options["SideLength if square (pixels)"]),
                                   tol=float(options["Root Found Tolerance"]))
    def AttractorFrac(self):
        options=self.AllOptions["Attractor"]
        
  
    def createandlabelentry(self,label,defaultval,FracType):#short hand for adding text boxes
        #prolly a simpler way but seems to work fine
        CurFrame=self.frames[FracType]
        OptionCount=len(self.AllOptions[FracType])*2
        print(len(self.AllOptions[FracType]))
        print(self.AllOptions)
        Entrywindow= tk.Entry(CurFrame)
        Entrywindowlabel=tk.Label(CurFrame,text=label)
        Entrywindow.insert(-1,str(defaultval))
        Entrywindow.grid(row=OptionCount+2,column=0,padx=2)
        Entrywindowlabel.grid(row=OptionCount+1,column=0,padx=2)
        self.AllOptions[FracType][label]=Entrywindow
        #print(self.AllOptions)        
        
        
        
    def __init__(self):
        
        #Create Panedwindow  
        self.panedwindow=ttk.Panedwindow(frame, orient=HORIZONTAL)  
        self.panedwindow.pack(fill=BOTH, expand=True)  
        #Create Frames  
        self.StabilityFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)  
        self.NewtonsFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)
        self.AttractorFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)  
        self.panedwindow.add(self.StabilityFracframe, weight=1)  
        self.panedwindow.add(self.NewtonsFracframe, weight=1) 
        self.panedwindow.add(self.AttractorFracframe, weight=1)
        self.frames={"Stability":self.StabilityFracframe,"Newtons":self.NewtonsFracframe,"Attractor":self.AttractorFracframe}
        
        self.AllOptions={"Stability":{},"Newtons":{},"Attractor":{}}
        
        # Button Creation
        StabilityFracButton = tk.Button(self.StabilityFracframe,
                                text = "Stability Fractal", 
                                command = self.StabilityFrac)

        NewtonsFracButton = tk.Button(self.NewtonsFracframe,
                                text = "Newtons Fractal", 
                                command = self.NewtonsFrac)

        AttractorFracButton = tk.Button(self.AttractorFracframe,
                                text = "Attractor Fractal", 
                                command = self.AttractorFrac)
        StabilityFracButton.grid(row=0,column=0,padx=2)
        NewtonsFracButton.grid(row=0,column=0,padx=2)
        AttractorFracButton.grid(row=0,column=0,padx=2)  
        
M=mainwindow()
M.createandlabelentry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Stability")
M.createandlabelentry("Seed Function","x**2+c","Stability")
M.createandlabelentry("SideLength if square (pixels)",1028,"Stability")
M.createandlabelentry("Max Iteration/Depth",2056,"Stability") 
M.createandlabelentry("Cycle Detection",True,"Stability")
M.createandlabelentry("Colour from Iteration Count",True,"Stability")
M.createandlabelentry("Tolerance for cycle Detection","Auto","Stability")

M.createandlabelentry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Newtons")
M.createandlabelentry("Seed Function","x**3+1","Newtons")
M.createandlabelentry("SideLength if square (pixels)",1028,"Newtons")
M.createandlabelentry("Max Iteration/Depth",2056,"Newtons")
M.createandlabelentry("Root Found Tolerance",1e-16,"Newtons") 

M.createandlabelentry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Attractor")
M.createandlabelentry("X updateFunc in X,Y,K1,K2,K3,K4,K5","X**3+1","Attractor")
M.createandlabelentry("Y updateFunc in X,Y,K1,K2,K3,K4,K5","X**3+1","Attractor")
M.createandlabelentry("Constants in form K1,K2,K3,K4,K5","2,1,0,0,0","Attractor")
M.createandlabelentry("SideLength if square (pixels)",1028,"Attractor")

M.createandlabelentry("Max Iteration/Depth",3000,"Attractor") 



frame.mainloop()
