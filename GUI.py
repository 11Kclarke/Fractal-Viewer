import tkinter as tk 
from tkinter import *
from tkinter import ttk
from Main import DrawStabilityFractalOpencl,DrawNewtonsFractalOpencl  
# Top level window
frame = tk.Tk()
frame.title("TextBox Input")
frame.geometry('400x600')
# Function for getting Input
# from textbox and printing it 
# at label widget
class mainwindow:
    def StabilityFrac(self):
        options=self.AllOptions["Stability"]
        if options['Cycle Detection']:
            cycles=16
        else:
            cycles=0
        print(options['Max Iteration/Depth'])
        extent=options['Extent/Corners in form x1,x2,y1,y2'].split(",")
        extent=map(float,extent)
        DrawStabilityFractalOpencl(*extent,
                                   options["Seed Function"],
                                   maxdepth=int(options['Max Iteration/Depth']),
                                   cycles=cycles,
                                   ittcountcolouring =options['Colour from Iteration Count'],
                                   npoints=options["X SideLength (pixels)"])
        
    def NewtonsFrac(self):
        options=self.AllOptions["Newtons"]

    def AttractorFrac(self):
        options=self.AllOptions["Attractor"]
        
  
    def createandlabelentry(self,label,defaultval,FracType):
        CurFrame=self.frames[FracType]
        OptionCount=len(self.AllOptions[FracType])*2
        print(len(self.AllOptions[FracType]))
        print(self.AllOptions)
        Entrywindow= tk.Entry(CurFrame)
        Entrywindowlabel=tk.Label(CurFrame,text=label)
        Entrywindow.insert(-1,str(defaultval))
        Entrywindow.grid(row=OptionCount+2,column=0,padx=2)
        Entrywindowlabel.grid(row=OptionCount+1,column=0,padx=2)
        self.AllOptions[FracType][label]=defaultval
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
M.createandlabelentry("X SideLength (pixels)",2000,"Stability")
M.createandlabelentry("Y SideLength (pixels)",2000,"Stability") 
M.createandlabelentry("Max Iteration/Depth",3000,"Stability") 
M.createandlabelentry("Cycle Detection",True,"Stability")
M.createandlabelentry("Colour from Iteration Count",True,"Stability")

M.createandlabelentry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Newtons")
M.createandlabelentry("Seed Function","X**3+1","Newtons")
M.createandlabelentry("X SideLength (pixels)",2000,"Newtons")
M.createandlabelentry("Y SideLength (pixels)",2000,"Newtons") 
M.createandlabelentry("Max Iteration/Depth",3000,"Newtons") 

M.createandlabelentry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Attractor")
M.createandlabelentry("Seed Function","X**3+1","Attractor")
M.createandlabelentry("X SideLength (pixels)",2000,"Attractor")
M.createandlabelentry("Y SideLength (pixels)",2000,"Attractor") 
M.createandlabelentry("Max Iteration/Depth",3000,"Attractor") 



frame.mainloop()
