# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:49:43 2018

@author: Rodolfo Luis Tonoli
"""
import os

# Standard names
std_root = ['root', 'Hips']
std_hips = ['spine05','Hips']
std_spine = ['spine04', 'Spine']
std_spine1 = ['spine03','Spine1']
std_spine2 = ['spine02','Spine2']
std_spine3 = ['spine01','Spine3']
std_neck = ['neck01','Neck']
std_neck1 = ['neck02','Neck1']
std_neckup = ['neck03']
std_head = ['head','Head']
std_rShoulder = ['shoulder01.R','RightShoulder']
std_rArm = ['upperarm01.R', 'RightArm']
std_rForeArm = ['lowerarm01.R', 'RightForeArm']
std_rHand = ['wrist.R', 'RightHand']
std_rHandThumb1 = []
std_rHandMiddle1 = ['finger3-1.R','RightHandMiddle1']
std_lShoulder = ['shoulder01.L','LeftShoulder']
std_lArm = ['upperarm01.L', 'LeftArm']
std_lForeArm = ['lowerarm01.L', 'LeftForeArm']
std_lHand = ['wrist.L', 'LeftHand']
std_lHandThumb1 = []
std_lHandMiddle1 = ['finger3-1.L','LeftHandMiddle1']
std_rUpLeg = ['upperleg01.R','RightUpLeg']
std_rLeg = ['lowerleg01.R','RightLeg']
std_rFoot = ['foot.R','RightForeFoot']
std_rForeFoot = []
std_rToeBase = ['toe3-1.R','RightToeBase']
std_lUpLeg = ['upperleg01.L','LeftUpLeg']
std_lLeg = ['lowerleg01.L','LeftLeg']
std_lFoot = ['foot.L','LeftForeFoot']
std_lForeFoot = []
std_lToeBase = ['toe3-1.L','LeftToeBase']

std_alljoints = std_root + std_hips + std_spine + std_spine1 + std_spine2 + std_spine3 + std_neck + std_neck1 + std_head + std_rShoulder + std_rArm + std_rForeArm + std_rHand + std_rHandThumb1 + std_rHandMiddle1 + std_lShoulder + std_lArm + std_lForeArm + std_lHand + std_lHandThumb1 + std_lHandMiddle1 + std_rUpLeg + std_rLeg + std_rFoot + std_rForeFoot + std_rToeBase + std_lUpLeg + std_lLeg + std_lFoot + std_lForeFoot + std_lToeBase
std_validNames = [std_root, std_hips, std_spine, std_spine1, std_spine2, std_spine3, std_neck, std_neck1, std_head, std_rShoulder, std_rArm, std_rForeArm, std_rHand, std_rHandThumb1, std_rHandMiddle1, std_lShoulder, std_lArm, std_lForeArm, std_lHand, std_lHandThumb1, std_lHandMiddle1, std_rUpLeg, std_rLeg, std_rFoot, std_rForeFoot, std_rToeBase, std_lUpLeg, std_lLeg, std_lFoot, std_lForeFoot, std_lToeBase]
std_ViconNames = ["Root", "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightUpLeg", "RightLeg", "RightFoot", "RightForeFoot", "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftForeFoot", "LeftToeBase"]
std_ViconJoints = [std_root, std_hips, std_spine, std_spine1, std_spine2, std_spine3, std_neck, std_neck1, std_head, std_rShoulder, std_rArm, std_rForeArm, std_rHand, std_lShoulder, std_lArm, std_lForeArm, std_lHand, std_rUpLeg, std_rLeg, std_rFoot, std_rForeFoot, std_rToeBase, std_lUpLeg, std_lLeg, std_lFoot, std_lForeFoot, std_lToeBase]


class SkeletonMap:
    def __init__(self, animation, mapfile=None):
        self.animation = animation
        self.alljoints = []
        self.__getCompleteMapping(animation, mapfile)
        self.checkJoints()
        self.animation.skeletonmap = self

    #def __str__(self):



    def getJointsNoRoot(self):
        jointlist = [self.hips, self.spine, self.spine1, self.spine2, self.spine3, self.neck, self.neck1, self.head, self.lshoulder,self.larm, self.lforearm, self.lhand, self.rshoulder, self.rarm, self.rforearm, self.rhand, self.lupleg, self.llowleg, self.lfoot, self.rupleg, self.rlowleg, self.rfoot]
        return jointlist

    def getJointsNoRootHips(self):
        jointlist = [self.spine, self.spine1, self.spine2, self.spine3, self.neck, self.neck1, self.head, self.lshoulder,self.larm, self.lforearm, self.lhand, self.rshoulder, self.rarm, self.rforearm, self.rhand, self.lupleg, self.llowleg, self.lfoot, self.rupleg, self.rlowleg, self.rfoot]

        return jointlist

    def getJointsLimbsHead(self):
        jointlist = [self.neck, self.neck1, self.head, self.larm, self.lforearm, self.lhand, self.rarm, self.rforearm, self.rhand, self.lupleg, self.llowleg, self.lfoot, self.rupleg, self.rlowleg, self.rfoot]
        return jointlist


    def vecRForearm(self, frame):
        return self.rhand.getPosition(frame) - self.rforearm.getPosition(frame)

    def vecLForearm(self, frame):
        return self.lhand.getPosition(frame) - self.lforearm.getPosition(frame)

    def vecRArm(self, frame):
        return self.rforearm.getPosition(frame) - self.rarm.getPosition(frame)

    def vecLArm(self, frame):
        return self.lforearm.getPosition(frame) - self.larm.getPosition(frame)

    def vecRClavicle(self, frame):
        return self.rarm.getPosition(frame) - self.spine3.getPosition(frame)

    def vecLClavicle(self, frame):
        return self.larm.getPosition(frame) - self.spine3.getPosition(frame)

    def vecNeck(self, frame):
        return self.spine3.getPosition(frame) - self.head.getPosition(frame)

    def vecSpine(self, frame):
        return self.spine3.getPosition(frame) - self.hips.getPosition(frame)

    def vecLFemur(self, frame):
        return self.lupleg.getPosition(frame) - self.hips.getPosition(frame)

    def vecRFemur(self, frame):
        return self.rupleg.getPosition(frame) - self.hips.getPosition(frame)

    def vecLUpleg(self, frame):
        return self.llowleg.getPosition(frame) - self.lupleg.getPosition(frame)

    def vecRUpleg(self, frame):
        return self.rlowleg.getPosition(frame) - self.rupleg.getPosition(frame)

    def vecLLowleg(self, frame):
        return self.lfoot.getPosition(frame) - self.llowleg.getPosition(frame)

    def vecRLowleg(self, frame):
        return self.rfoot.getPosition(frame) - self.rlowleg.getPosition(frame)

    def checkJoints(self):
        #Note: Not all joints are being checked
        if not self.alljoints:
            print('Mapping not performed')
            return None
        flag = True
        if self.root == None:
            print('Root not found')
            flag = False
        if self.hips == None:
            print('Hips not found')
            flag = False
        if self.spine3 == None:
            print('Spine 3 not found')
            flag = False
        if self.spine2 == None:
            print('Spine 2 not found')
            flag = False
        if self.spine == None:
            print('Spine not found')
            flag = False
        if self.head == None:
            print('Head not found')
            flag = False
        if self.neck == None:
            print('Neck not found')
            flag = False
        if self.neck1 == None:
            print('Neck1 not found')
            flag = False
        if self.lshoulder == None:
            print('Left Shoulder not found')
            flag = False
        if self.lforearm == None:
            print('Left Forearm not found')
            flag = False
        if self.larm == None:
            print('Left Arm not found')
            flag = False
        if self.lhand == None:
            print('Left Hand not found')
            flag = False
        if self.lhandmiddle == None:
            print('Left Hand Middle not found')
            flag = False
        if self.rshoulder == None:
            print('Right Shoulder not found')
            flag = False
        if self.rforearm == None:
            print('Rigth Forearm not found')
            flag = False
        if self.rarm == None:
            print('Right Arm not found')
            flag = False
        if self.rhand == None:
            print('Right Hand not found')
            flag = False
        if self.rhandmiddle == None:
            print('Right Hand Middle not found')
            flag = False
        if self.lupleg == None:
            print('Left Upper Leg not found')
            flag = False
        if self.llowleg == None:
            print('Left Lower Leg found')
            flag = False
        if self.lfoot == None:
            print('Left Foot not found')
            flag = False
        if self.ltoebase == None:
            print('Left Toe Base not found')
            flag = False
        if self.rupleg == None:
            print('Right Upper Leg not found')
            flag = False
        if self.rlowleg == None:
            print('Right Leg not found')
            flag = False
        if self.rfoot == None:
            print('Right Foot not found')
            flag = False
        if self.rtoebase == None:
            print('Right Toe Base not found')
            flag = False
        if flag:
            print('The mapping was successful')
        else:
            return None


    def __getCompleteMapping(self, animation, mapfile):
        self.root = None
        # Spine
        self.hips, self.spine, self.spine1, self.spine2, self.spine3 = None, None, None, None, None
        self.head, self.neck, self.neck1 = None, None, None

        self.lshoulder, self.rshoulder = None, None

        # Upper limbs
        self.larm, self.lforearm, self.lhand, self.lhandmiddle, self.rarm, self.rforearm, self.rhand, self.rhandmiddle = None, None, None, None, None, None, None, None
        # Lower limbs
        self.lupleg, self.llowleg, self.lfoot, self.ltoebase, self.rupleg, self.rlowleg, self.rfoot, self.rtoebase = None, None, None, None, None, None, None, None


        # Not on the list:
        self.neckup, self.lforefoot, self.rforefoot, self.lhandthumb, self.rhandthumb = None, None, None, None, None

        if mapfile:
            realpath = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(realpath, mapfile)
            try:
                with open(path, "r") as file:
                    data = [line.replace('\n', '').split(',') for line in file]
                    # Custom names
                    root = [data[0][1]]
                    hips = [data[1][1]]
                    spine = [data[2][1]]
                    spine1 = [data[3][1]]
                    spine2 = [data[4][1]]
                    spine3 = [data[5][1]]
                    neck = [data[6][1]]
                    neck1 = [data[7][1]]
                    neckup = [data[8][1]]
                    head = [data[9][1]]
                    rShoulder = [data[10][1]]
                    rArm = [data[11][1]]
                    rForeArm = [data[12][1]]
                    rHand = [data[13][1]]
                    rHandThumb1 = [data[14][1]]
                    rHandMiddle1 = [data[15][1]]
                    lShoulder = [data[16][1]]
                    lArm = [data[17][1]]
                    lForeArm = [data[18][1]]
                    lHand = [data[19][1]]
                    lHandThumb1 = [data[20][1]]
                    lHandMiddle1 = [data[21][1]]
                    rUpLeg = [data[22][1]]
                    rLeg = [data[23][1]]
                    rFoot = [data[24][1]]
                    rForeFoot = [data[25][1]]
                    rToeBase = [data[26][1]]
                    lUpLeg = [data[27][1]]
                    lLeg = [data[28][1]]
                    lFoot = [data[29][1]]
                    lForeFoot = [data[30][1]]
                    lToeBase = [data[31][1]]
            except FileNotFoundError as e:
                print('Invalid path provided or file not found in %s.\nError: %s.' % (realpath,str(e)))
                return None
        else:
            root,hips,spine,spine1,spine2,spine3,neck,neck1,neckup,head,rShoulder,rArm,rForeArm ,rHand,rHandThumb1 ,rHandMiddle1,lShoulder,lArm,lForeArm ,lHand,lHandThumb1,lHandMiddle1 ,rUpLeg,rLeg,rFoot,rForeFoot,rToeBase,lUpLeg,lLeg,lFoot,lForeFoot,lToeBase = std_root,std_hips,std_spine,std_spine1,std_spine2,std_spine3,std_neck,std_neck1,std_neckup,std_head ,std_rShoulder,std_rArm,std_rForeArm ,std_rHand,std_rHandThumb1 ,std_rHandMiddle1,std_lShoulder,std_lArm,std_lForeArm ,std_lHand,std_lHandThumb1,std_lHandMiddle1 ,std_rUpLeg,std_rLeg,std_rFoot,std_rForeFoot,std_rToeBase,std_lUpLeg,std_lLeg,std_lFoot,std_lForeFoot,std_lToeBase

        for joint in animation.getlistofjoints():
            if isroot(joint.name,root):
                self.root = joint
                if iships(joint.name, root):
                    self.hips = joint
            # Spine
            if iships(joint.name, hips):
                self.hips = joint
            if isspine3(joint.name, spine3):
                self.spine3 = joint
            if isspine2(joint.name, spine2):
                self.spine2 = joint
            if isspine1(joint.name, spine1):
                self.spine1 = joint
            if isspine(joint.name, spine):
                self.spine = joint

            # Head
            elif ishead(joint.name, head):
                self.head = joint
            elif isneck(joint.name, neck):
                self.neck = joint
            elif isneck1(joint.name, neck1):
                self.neck1 = joint

            # Upper limbs
            elif isleftshoulder(joint.name, lShoulder):
                self.lshoulder = joint
            elif isleftarm(joint.name, lArm):
                self.larm = joint
            elif isleftforearm(joint.name, lForeArm):
                self.lforearm= joint
            elif islefthand(joint.name, lHand):
                self.lhand = joint
            elif isrightshoulder(joint.name, rShoulder):
                self.rshoulder = joint
            elif isrightarm(joint.name, rArm):
                self.rarm = joint
            elif isrightforearm(joint.name, rForeArm):
                self.rforearm = joint
            elif isrighthand(joint.name, rHand):
                self.rhand = joint

            # Lower limbs
            elif isleftupleg(joint.name, lUpLeg):
                self.lupleg = joint
            elif isleftleg(joint.name, lLeg):
                self.llowleg = joint
            elif isleftfoot(joint.name, lFoot):
                self.lfoot = joint
            elif isleftforefoot(joint.name, lForeFoot):
                self.lforefoot = joint
            elif islefttoebase(joint.name, lToeBase):
                self.ltoebase = joint
            elif isrightupleg(joint.name, rUpLeg):
                self.rupleg = joint
            elif isrightleg(joint.name, rLeg):
                self.rlowleg = joint
            elif isrightfoot(joint.name, rFoot):
                self.rfoot = joint
            elif isrightforefoot(joint.name, rForeFoot):
                self.rforefoot = joint
            elif isrighttoebase(joint.name, rToeBase):
                self.rtoebase = joint

            # Not added to list:
            elif isrighthandmiddle1(joint.name, rHandMiddle1):
                self.rhandmiddle = joint
            elif islefthandmiddle1(joint.name, lHandMiddle1):
                self.lhandmiddle = joint

        self.alljoints = [self.root,
                          self.hips,
                          self.spine,
                          self.spine1,
                          self.spine2,
                          self.spine3,
                          self.neck,
                          self.neck1,
                          self.neckup,
                          self.head,
                          self.lshoulder,
                          self.larm,
                          self.lhandthumb,
                          self.lhandmiddle,
                          self.lforearm,
                          self.lhand,
                          self.rshoulder,
                          self.rarm,
                          self.rhandthumb,
                          self.rhandmiddle,
                          self.rforearm,
                          self.rhand,
                          self.lupleg,
                          self.llowleg,
                          self.lfoot,
                          self.lforefoot,
                          self.ltoebase,
                          self.rupleg,
                          self.rlowleg,
                          self.rfoot,
                          self.rforefoot,
                          self.rtoebase]


def isroot(jointname, root=std_root):
    if any(name == jointname for name in root):
        return root
    else:
        return None

def iships(jointname, hips=std_hips):
    if any(name == jointname for name in hips):
        return hips
    else:
        return None

def isspine(jointname, spine=std_spine):
    if any(name == jointname for name in spine):
        return spine
    else:
        return None

def isspine1(jointname, spine1=std_spine1):
    if any(name == jointname for name in spine1):
        return spine1
    else:
        return None

def isspine2(jointname, spine2=std_spine2):
    if any(name == jointname for name in spine2):
        return spine2
    else:
        return None

def isspine3(jointname, spine3=std_spine3):
    if any(name == jointname for name in spine3):
        return spine3
    else:
        return None

def isneck(jointname, neck=std_neck):
    if any(name == jointname for name in neck):
        return neck
    else:
        return None

def isneck1(jointname, neck1=std_neck1):
    if any(name == jointname for name in neck1):
        return neck1
    else:
        return None

def isneckup(jointname, neckup=std_neckup):
    if any(name == jointname for name in neckup):
        return neckup
    else:
        return None

def ishead(jointname, head=std_head):
    if any(name == jointname for name in head):
        return head
    else:
        return None

def isrightshoulder(jointname, rShoulder=std_rShoulder):
    if any(name == jointname for name in rShoulder):
        return rShoulder
    else:
        return None

def isrightarm(jointname, rArm=std_rArm):
    if any(name == jointname for name in rArm):
        return rArm
    else:
        return None

def isrightforearm(jointname, rForeArm=std_rForeArm):
    if any(name == jointname for name in rForeArm):
        return rForeArm
    else:
        return None

def isrighthand(jointname, rHand=std_rHand):
    if any(name == jointname for name in rHand):
        return rHand
    else:
        return None


def isrighthandthumb1(jointname, rHandThumb1=std_rHandThumb1):
    if any(name == jointname for name in rHandThumb1):
        return rHandThumb1
    else:
        return None


def isrighthandmiddle1(jointname, rHandMiddle1=std_rHandMiddle1):
    if any(name == jointname for name in rHandMiddle1):
        return rHandMiddle1
    else:
        return None


def isleftshoulder(jointname, lShoulder=std_lShoulder):
    if any(name == jointname for name in lShoulder):
        return lShoulder
    else:
        return None


def isleftarm(jointname, lArm=std_lArm):
    if any(name == jointname for name in lArm):
        return lArm
    else:
        return None


def isleftforearm(jointname, lForeArm=std_lForeArm):
    if any(name == jointname for name in lForeArm):
        return lForeArm
    else:
        return None


def islefthand(jointname, lHand=std_lHand):
    if any(name == jointname for name in lHand):
        return lHand
    else:
        return None


def islefthandthumb1(jointname, lHandThumb1=std_lHandThumb1):
    if any(name == jointname for name in lHandThumb1):
        return lHandThumb1
    else:
        return None


def islefthandmiddle1(jointname, lHandMiddle1=std_lHandMiddle1):
    if any(name == jointname for name in lHandMiddle1):
        return lHandMiddle1
    else:
        return None


def isrightupleg(jointname, rUpLeg=std_rUpLeg):
    if any(name == jointname for name in rUpLeg):
        return rUpLeg
    else:
        return None


def isrightleg(jointname, rLeg=std_rLeg):
    if any(name == jointname for name in rLeg):
        return rLeg
    else:
        return None


def isrightfoot(jointname, rFoot=std_rFoot):
    if any(name == jointname for name in rFoot):
        return rFoot
    else:
        return None


def isrightforefoot(jointname, rForeFoot=std_rForeFoot):
    if any(name == jointname for name in rForeFoot):
        return rForeFoot
    else:
        return None


def isrighttoebase(jointname, rToeBase=std_rToeBase):
    if any(name == jointname for name in rToeBase):
        return rToeBase
    else:
        return None


def isleftupleg(jointname, lUpLeg=std_lUpLeg):
    if any(name == jointname for name in lUpLeg):
        return lUpLeg
    else:
        return None


def isleftleg(jointname, lLeg=std_lLeg):
    if any(name == jointname for name in lLeg):
        return lLeg
    else:
        return None


def isleftfoot(jointname, lFoot=std_lFoot):
    if any(name == jointname for name in lFoot):
        return lFoot
    else:
        return None


def isleftforefoot(jointname, lForeFoot=std_lForeFoot):
    if any(name == jointname for name in lForeFoot):
        return lForeFoot
    else:
        return None


def islefttoebase(jointname, lToeBase=std_lToeBase):
    if any(name == jointname for name in lToeBase):
        return lToeBase
    else:
        return None


def isany(jointname, alljoints=std_alljoints):
    if any(name == jointname for name in alljoints):
        return True
    else:
        return False


# TODO: Resolver
def whichis(jointname):
    """
    Returns the method that the joint passed

    :type jointname: str
    :param jointname: Name of the mapped joint
    """
    cases = [isroot,iships,isspine,isspine1,isspine2,isspine3,isneck,isneck1,isneckup,ishead,isrightshoulder,isrightarm,isrightforearm,isrighthand,isrighthandthumb1,isrighthandmiddle1,isleftshoulder,isleftarm,isleftforearm,islefthand,islefthandthumb1,islefthandmiddle1,isrightupleg,isrightleg,isrightfoot,isrightforefoot,isrighttoebase,isleftupleg,isleftleg,isleftfoot,isleftforefoot,islefttoebase]
    for case in cases:
        isthis = case(jointname)
        if isthis:
            return case, cases


# TODO: Resolver
def ishand(jointname):
    if islefthand(jointname) or isrighthand(jointname):
        return True
    else:
        return False


# TODO: Resolver
def getmatchingjoint(jointname, animation, returnnear=True):
    """
    Get the list of names that contains jointname and search for a joint in the
    animation using the names on the list

    :type jointname: str
    :param jointname: Name of the joint

    :type animation: pyanimation.Animation
    :param animation: Animation to look for a mapped joint
    """
    passedcase, cases = whichis(jointname)
    custommap = [joint.name if joint else None for joint in animation.skeletonmap.alljoints]
    ind = cases.index(passedcase)
    joint = animation.getJoint(custommap[ind])
    #if not joint and ((jointname == 'Spine1') or (jointname == 'Spine'))
    assert joint, str.format('There is not a mapped joint in the animation that matches with the joint %s' % jointname)
    return joint


# TODO: Resolver
def isamatch(jointname1, jointname2, validNames=std_validNames):
    for validnamesofONEjoint in validNames:
        if any(name == jointname1 for name in validnamesofONEjoint):
            if any(name == jointname2 for name in validnamesofONEjoint):
                return True
    return False


def getIntermediateSkeleton(animation):
    # Should no be used
    print('Do not use this method, create a SkeletonMap object instead.')
    root, hips, spine, spine1, spine2, spine3, head, larm, lforearm, lhand, rarm, rforearm, rhand = None,None,None,None,None,None,None,None,None,None,None,None,None
    for joint in animation.getlistofjoints():
        if isroot(joint.name):
            root = joint
        elif iships(joint.name):
            hips = joint
        elif isspine3(joint.name):
            spine3 = joint
        elif isspine2(joint.name):
            spine2 = joint
        elif isspine1(joint.name):
            spine1 = joint
        elif isspine(joint.name):
            spine = joint
        elif ishead(joint.name):
            head = joint
        elif isleftforearm(joint.name):
            lforearm= joint
        elif isleftarm(joint.name):
            larm = joint
        elif islefthand(joint.name):
            lhand = joint
        elif isrightforearm(joint.name):
            rforearm = joint
        elif isrightarm(joint.name):
            rarm = joint
        elif isrighthand(joint.name):
            rhand = joint
    found = 1
    missed = 0
    for joint in [root, hips,spine, spine1, spine2, spine3, head, larm, lforearm, lhand, rarm, rforearm, rhand]:
        if joint:
            found = found + 1
        else:
            missed = missed + 1
    #print('%d out of %d joints found' % (found, found+missed))
    return root, hips, spine, spine1, spine2, spine3, head, larm, lforearm, lhand, rarm, rforearm, rhand


# allLimbsNames = rArmNames + rForeArmNames + rHandNames + lArmNames + lForeArmNames + lHandNames + lUpperLegNames + lLegNames + lFootNames + rUpperLegNames + rLegNames + rFootNames
# upperLimbsNames = rArmNames + rForeArmNames + rHandNames + lArmNames + lForeArmNames + lHandNames
# upperExtremitiesNames = rArmNames + lArmNames
