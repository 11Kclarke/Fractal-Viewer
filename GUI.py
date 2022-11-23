import tkinter as tk 
from tkinter import *
from tkinter import ttk
from FrontEnd import DrawStabilityFractalOpencl,DrawNewtonsFractalOpencl,DrawJuliaFractalOpencl,Draw2axisJuliaStabilityFractalOpencl
from pyopenclAttractorFractals import AttractorExplorer
import numpy as np
# Top level window
frame = tk.Tk()
frame.title("TextBox Input")
frame.geometry('1400x800')

class mainwindow:
    def getOptionsStability(self,options):
        for i in options.keys():
            try:
                options[i]=options[i].get()
            except AttributeError:
                options[i]=options[i]["text"]
        if not (options['Cycle Detection'].lower() in ["false","n","no","0"]):
            cycles=10
        else:
            cycles=0         
        Seedfunc=options["Seed Function"].lower()         
        if options["Tolerance for cycle Detection"].lower() != "auto": 
            try:
                cycleacc=float(options["Tolerance for cycle Detection"])
            except ValueError:
                cycleacc=None
                print("invalid cycle Accuracy, defaulting to auto")
        else:cycleacc=None
        if options["Algorithm variations"]=="Burning Ship":
            variation="Burning Ship"
        elif options["Algorithm variations"]=="Tricorn":
            variation="Tricorn"
        else:
            variation=""    
        extent=options['Extent/Corners in form x1,x2,y1,y2'].split(",")
        extent=list(map(float,extent))
        if options['Colour from Iteration Count'].lower() in ["false","n","no","0"]:
            ittcountcolouring=False
        else:
            ittcountcolouring=True
        if options["Use Custom Input"] == "False":
            if options["Use Preselected Function"]=="Reciprocal Mandlebrot":
                Seedfunc="1/(x**2+c)"
                if options["Use Preselected Args With Preselected Function"]=="True":
                    extent=[1,-0.5,1,-0.5]
        #x1,x2,y1,y2,fl,npoints=1024, maxdepth=3000,cycles=16,cycleacc=None,ittcountcolouring=True,Divlim=2,variation="",ShowOrbits=True
        return(extent,Seedfunc,variation,cycleacc,ittcountcolouring,cycles,options)
    def StabilityFrac(self,julia=False):
        
        options=self.AllOptions["Stability"].copy()
        extent,Seedfunc,variation,cycleacc,ittcountcolouring,cycles,options=self.getOptionsStability(options)
        DrawStabilityFractalOpencl(*extent,
                                   Seedfunc,
                                   maxdepth=int(options['Max Iteration/Depth']),
                                   cycles=cycles,
                                   ittcountcolouring =ittcountcolouring,
                                   npoints=int(options["SideLength if square (pixels)"]),
                                   cycleacc=cycleacc,
                                   Divlim=float(options["Divergence Limit"]),
                                   variation=variation)
    def JuliaFrac(self):
        options=self.AllOptions["Julia"].copy()
        extent,Seedfunc,variation,cycleacc,ittcountcolouring,cycles,options=self.getOptionsStability(options)
        C=options["Complex Seed Constant"].replace(" ","")
        C=complex(C)
        if options["Plot Corresponding Stability fractal\n and Use to Explore Different Seed Constants"]:
            Draw2axisJuliaStabilityFractalOpencl(*extent,
                                                C,
                                                Seedfunc,
                                                maxdepth=int(options['Max Iteration/Depth']),
                                                cycles=cycles,
                                                ittcountcolouring =ittcountcolouring,
                                                npoints=int(options["SideLength if square (pixels)"]),
                                                cycleacc=cycleacc,
                                                Divlim=float(options["Divergence Limit"]),
                                                variation=variation)
        else:
            DrawJuliaFractalOpencl(*extent,
                                    C,
                                    Seedfunc,
                                    maxdepth=int(options['Max Iteration/Depth']),
                                    cycles=cycles,
                                    ittcountcolouring =ittcountcolouring,
                                    npoints=int(options["SideLength if square (pixels)"]),
                                    cycleacc=cycleacc,
                                    Divlim=float(options["Divergence Limit"]),
                                    variation=variation)
          
    def NewtonsFrac(self):
        options=self.AllOptions["Newtons"].copy()
        for i in options.keys():
            try:
                options[i]=options[i].get()
            except AttributeError:
                options[i]=options[i]["text"]
        extent=options['Extent/Corners in form x1,x2,y1,y2'].split(",")
        extent=list(map(float,extent))
        DrawNewtonsFractalOpencl(*extent,
                                   options["Seed Function"].lower(),
                                   maxdepth=int(options['Max Iteration/Depth']),
                                   npoints=int(options["SideLength if square (pixels)"]),
                                   tol=float(options["Root Found Tolerance"]))
    def AttractorFrac(self):
        options=self.AllOptions["Attractor"].copy()
        for i in options.keys():
            try:
                options[i]=options[i].get()
            except AttributeError:
                options[i]=options[i]["text"]
        args = options["Constants in form K1,K2,K3,K4,K5"].split(",")
        args = list(map(float,args))
        SideLength=int(options["SideLength if square (pixels)"])
        IttLim=int(options['Max Iteration/Depth'])
        Res2=int(options["Res of Orbit"])
        N2=int(float(options["Orbit IttLim"]))
        extent=options['Extent/Corners in form x1,x2,y1,y2'].split(",")
        extent=list(map(float,extent))
        if options["Use Custom Input"] == "False":
            if options["Use Preselected Function"]=="Gumowski_Mira":
                print("Gumowski_Mira")
                if options["Use Preselected Args With Preselected Function"]=="True":
                    args = [1,np.cos(4*np.pi/5)+0.008,  0.01, 0, 0]
                    SideLength = 500
                    IttLim=50
                    extent=[-2,2,-2,2]
                    Res2=500
                    N2=int(1e6)
                Gumowski_MiraF = "(k2 * x + (2*(1-k2)*x*x/(1.0 + x * x )))"
                Gumowski_Mirax="k1*y+k3* y * (1-k3* y * y )+"+Gumowski_MiraF
                Gumowski_Miray="-x+"+Gumowski_MiraF.replace("x","XN")
                AttractorExplorer(*extent,
                                  IttLim,
                                  Gumowski_Mirax,
                                  Gumowski_Miray,
                                  SideLength,
                                  Res2=Res2,
                                  N2=N2,
                                  args=args)
            elif options["Use Preselected Function"]=="Hoppalong":
                    if options["Use Preselected Args With Preselected Function"]=="True":
                        print("Hoppalong")
                        args = [1.1,0.5,  1, 0, 0]
                        SideLength = 500
                        IttLim=50
                        extent=[-2,2,-2,2]
                        Res2=500
                        N2=int(1e6)
                    hoppalongx = "Y[i]-sign( X[i] )*( sqrt(fabs(k2* X[i] -k3)))"
                    hoppalongy = "k1- X[i] "
                    AttractorExplorer(*extent,
                                    IttLim,
                                    hoppalongx,
                                    hoppalongy,
                                    SideLength,
                                    Res2=Res2,
                                    N2=N2,
                                    args=args)
        else:
              
            AttractorExplorer(*extent,
                            IttLim,
                            options["X updateFunc in X,Y,K1,K2,K3,K4,K5,X0,Y0"].lower(),
                            options["Y updateFunc in X,Y,K1,K2,K3,K4,K5,XN,Y0"].lower(),
                            SideLength,Res2=Res2,N2=N2,args=args)
        
        """ AttractorExplorer(*extent,
                          int(options['Max Iteration/Depth']),
                          options["X updateFunc in X,Y,K1,K2,K3,K4,K5,X0,Y0"],
                          options["Y updateFunc in X,Y,K1,K2,K3,K4,K5,XN,Y0"],
                          int(options["SideLength if square (pixels)"]),
                          Res2=int(options["Res of Orbit"]),
                          N2=int(options["Orbit IttLim"]),
                          args=options["Constants in form K1,K2,K3,K4,K5"])"""
                        
  
    def CreateAndLabelSwitch(self,label,defaultval,FracType):#short hand for adding text boxes
        #prolly a simpler way but seems to work fine
        CurFrame=self.frames[FracType]
        OptionCount=len(self.AllOptions[FracType])*2
        def switch():#function ran on button press
            if self.AllOptions[FracType][label]["text"].lower()=="true":
                self.AllOptions[FracType][label].config(bg="red")
                self.AllOptions[FracType][label]["text"]="False"   
            else:
                self.AllOptions[FracType][label].config(bg="green")
                self.AllOptions[FracType][label]["text"]="True"
        Button= tk.Button(CurFrame,command=switch)
        Buttonlabel=tk.Label(CurFrame,text=label)
        Button.config(text=str(defaultval))
        if defaultval=="True":Button.config(bg="green")#start colour
        else:Button.config(bg="red")
        Button.grid(row=OptionCount+2,column=0,pady=2,columnspan=1)
        Buttonlabel.grid(row=OptionCount+1,column=0)
        
        self.AllOptions[FracType][label]=Button
        
        
        #print(self.AllOptions)        
    def CreateAndLabelEntry(self,label,defaultval,FracType):#short hand for adding text boxes
        #prolly a simpler way but seems to work fine
       
        CurFrame=self.frames[FracType]
        OptionCount=(len(self.AllOptions[FracType]))*2
        
        Entrywindow= tk.Entry(CurFrame,width=40)
        Entrywindowlabel=tk.Label(CurFrame,text=label)
        Entrywindow.insert(-1,str(defaultval))
        Entrywindow.grid(row=OptionCount+2,column=0,padx=2)
        Entrywindowlabel.grid(row=OptionCount+1,column=0,padx=2)
        self.AllOptions[FracType][label]=Entrywindow
        #print(self.AllOptions)            
    
    def CreateAndLabelDropdown(self,label,defaultval,FracType,Options):#short hand for adding text boxes
        #prolly a simpler way but seems to work fine
        CurFrame=self.frames[FracType]
        OptionCount=len(self.AllOptions[FracType])*2
        #print(len(self.AllOptions[FracType]))
        #print(self.AllOptions)
        
        Dropdown= tk.OptionMenu(CurFrame,tk.StringVar(CurFrame,defaultval),defaultval,*Options)
        Dropdownlabel=tk.Label(CurFrame,text=label)
        #Dropdown.insert(-1,str(defaultval))
        Dropdown.grid(row=OptionCount+2,column=0,pady=2)
        Dropdownlabel.grid(row=OptionCount+1,column=0,padx=2)
        self.AllOptions[FracType][label]=Dropdown    
        #print(OptionCount)
    def __init__(self):
        
        #Create Panedwindow  
        self.panedwindow=ttk.Panedwindow(frame, orient=HORIZONTAL)  
        self.panedwindow.pack(fill=BOTH, expand=True)  
        #Create Frames  
        self.StabilityFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)
        self.JuliaFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)  
        self.NewtonsFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)
        self.AttractorFracframe=ttk.Frame(self.panedwindow,width=400,height=400, relief=SUNKEN)  
        self.panedwindow.add(self.StabilityFracframe, weight=1)
        self.panedwindow.add(self.JuliaFracframe, weight=1)  
        self.panedwindow.add(self.NewtonsFracframe, weight=1) 
        self.panedwindow.add(self.AttractorFracframe, weight=1)
        self.frames={"Stability":self.StabilityFracframe,"Julia":self.JuliaFracframe,"Newtons":self.NewtonsFracframe,"Attractor":self.AttractorFracframe}
        
        self.AllOptions={"Stability":{},"Julia":{},"Newtons":{},"Attractor":{}}
        
        # Button Creation
        StabilityFracButton = ttk.Button(self.StabilityFracframe,
                                text = "Stability Fractal", 
                                command = self.StabilityFrac)
        
        JuliaFracButton = ttk.Button(self.JuliaFracframe,
                                text = "Julia Fractal", 
                                command = self.JuliaFrac)

        NewtonsFracButton = ttk.Button(self.NewtonsFracframe,
                                text = "Newtons Fractal", 
                                command = self.NewtonsFrac)

        AttractorFracButton = ttk.Button(self.AttractorFracframe,
                                text = "Attractor Fractal", 
                                command = self.AttractorFrac)
        StabilityFracButton.grid(row=0,column=0,padx=0,pady=0)
        JuliaFracButton.grid(row=0,column=0,padx=0,pady=0)
        NewtonsFracButton.grid(row=0,column=0,padx=2)
        AttractorFracButton.grid(row=0,column=0,padx=2)  
        
M=mainwindow()

M.CreateAndLabelSwitch("Use Custom Input","True","Stability")
M.CreateAndLabelDropdown("Algorithm variations","Standard Mandlebrot like","Stability",["Burning Ship","Tricorn"])
M.CreateAndLabelDropdown("Use Preselected Function","MandleBrot","Stability",["Reciprocal Mandlebrot","more to be added"])
M.CreateAndLabelSwitch("Use Preselected Args With Preselected Function","True","Stability")
M.CreateAndLabelEntry("Extent/Corners in form x1,x2,y1,y2","-2,2,-2,2","Stability")
M.CreateAndLabelEntry("Seed Function","x**2+c","Stability")
M.CreateAndLabelEntry("SideLength if square (pixels)",1028,"Stability")
M.CreateAndLabelEntry("Max Iteration/Depth",2056,"Stability") 
M.CreateAndLabelEntry("Cycle Detection",True,"Stability")
M.CreateAndLabelEntry("Colour from Iteration Count",True,"Stability")
M.CreateAndLabelEntry("Tolerance for cycle Detection","Auto","Stability")
M.CreateAndLabelEntry("Divergence Limit","2","Stability")

M.CreateAndLabelSwitch("Plot Corresponding Stability fractal\n and Use to Explore Different Seed Constants","True","Julia")
M.CreateAndLabelSwitch("Use Custom Input","True","Julia")
M.CreateAndLabelDropdown("Algorithm variations","Standard Mandlebrot like","Julia",["Burning Ship","Tricorn"])
M.CreateAndLabelDropdown("Use Preselected Function","MandleBrot","Julia",["Reciprocal Mandlebrot","more to be added"])
M.CreateAndLabelSwitch("Use Preselected Args With Preselected Function","True","Julia")
M.CreateAndLabelEntry("Extent/Corners in form x1,x2,y1,y2","-2,2,-2,2","Julia")
M.CreateAndLabelEntry("Complex Seed Constant","-0.4+0.6j","Julia")
M.CreateAndLabelEntry("Seed Function","x**2+c","Julia")
M.CreateAndLabelEntry("SideLength if square (pixels)",1028,"Julia")
M.CreateAndLabelEntry("Max Iteration/Depth",2056,"Julia") 
M.CreateAndLabelEntry("Cycle Detection",True,"Julia")
M.CreateAndLabelEntry("Colour from Iteration Count",True,"Julia")
M.CreateAndLabelEntry("Tolerance for cycle Detection","Auto","Julia")
M.CreateAndLabelEntry("Divergence Limit","2","Julia")

M.CreateAndLabelSwitch("Use Custom Input","True","Newtons")
M.CreateAndLabelDropdown("Use Preselected Function","X**3+1","Newtons",["Exponential","more to be added"])
M.CreateAndLabelEntry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Newtons")
M.CreateAndLabelEntry("Seed Function","x**3+1","Newtons")
M.CreateAndLabelEntry("SideLength if square (pixels)",1028,"Newtons")
M.CreateAndLabelEntry("Max Iteration/Depth",2056,"Newtons")
M.CreateAndLabelEntry("Root Found Tolerance",1e-16,"Newtons") 

M.CreateAndLabelSwitch("Use Custom Input","True","Attractor")
M.CreateAndLabelDropdown("Use Preselected Function","Hoppalong","Attractor",["Gumowski_Mira","more to be added"])
M.CreateAndLabelSwitch("Use Preselected Args With Preselected Function","True","Attractor")
M.CreateAndLabelEntry("Extent/Corners in form x1,x2,y1,y2","2,-2,2,-2","Attractor")
M.CreateAndLabelEntry("X updateFunc in X,Y,K1,K2,K3,K4,K5,X0,Y0","Y-sign(X)*(sqrt(fabs(k2*X-k3)))","Attractor")
M.CreateAndLabelEntry("Y updateFunc in X,Y,K1,K2,K3,K4,K5,XN,Y0","k1- X","Attractor")
M.CreateAndLabelEntry("Constants in form K1,K2,K3,K4,K5","2,1,0,0,0","Attractor")
M.CreateAndLabelEntry("SideLength if square (pixels)",500,"Attractor")
M.CreateAndLabelEntry("Res of Orbit",300,"Attractor")
M.CreateAndLabelEntry("Orbit IttLim",1e6,"Attractor")
M.CreateAndLabelEntry("Max Iteration/Depth",50,"Attractor") 



frame.mainloop()
