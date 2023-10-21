#!/usr/bin/env python3
#_∗_coding: utf-8 _∗_


"""
"""
import os
import io
import string
import re
import sys
import datetime
import copy
import json
import operator #数学计算操作符
import random
import time
import matplotlib.pyplot as plt
import pylab as mpl     #import matplotlib as mpl
#import networkx as nx
#import pydot
#from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import cPickle as pickle
except ImportError:
    import pickle

#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
#mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体 FangSong,SimHei
#mpl.rcParams['font.serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.append("D:\\TexasHoldem\\LuaDemo\\agent") 
sys.path.append("D:\\TexasHoldem\\TexasAI") 
from Ccardev.evaluatordll import *
print('sys.path=',sys.path)

colorline=['-','--','+-','-.','y','m1']
colors_use=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
    '#bec1d4', '#bb7784', '#0000ff', '#111010', '#FFFF00', '#1f77b4', '#800080',
    '#959595', '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b', '#4a6fe3',
    '#8595e1', '#b5bbe3', '#e6afb9', '#e07b91', '#d33f6a', '#11c638', '#8dd593',
    '#c6dec7', '#ead3c6', '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7',
    '#f3e1eb', '#f6c4e1', '#f79cd4']
markline=['o','v','<','>','1','2']





# ACPCserver平台给出的结果文件的处理
datausers=[]
recAllresults=[]
recAllagentsname=[]
#第三个参数flagsinglefile是一个标记处理单文件的标记，若为true，收集的数据仅是当前给定文件的
#若非flase那么是集合多个文件的。
def logfiledealing(filename,myname,flagsinglefile=True,bigblindfirst=False,flagfoldseen=True):
    global datausers
    global recAllresults
    global recAllagentsname
    
    if flagsinglefile:
        recAllresults=[]

    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    lino=0
    for r in resdata:
        lino+=1
        if (r.count("STATE")>0):
            data=ACPClogmsgTodata(r.strip(),myname,bigblindfirst,flagfoldseen)
            #print('data=',data)
            recAllresults.append(data)

    #记录所有玩家的名称信息
    recAllagentsname=[]
    for player in recAllresults[0]['players']:
        if player['name'] not in recAllagentsname:
            recAllagentsname.append(player['name'])
    print('recAllagentsname=',recAllagentsname)

    datausers=recAllresults
    return None




# 将acpc的log的信息转换为data数据
# 注意这是两人情况下用的，因为其中很多处理都是针对两人做的。
# 输入是log文件中的字符串和玩家的名字
# BigBlindfirst选项用于控制大盲注先行的情况，从acpc收集的数据看是有大盲注线性的。
# 而默认是小盲注先的。
params_sb = 50
params_bb = 100
params_stack = 20000
params_maxr = 3  #一轮最大raise次数
def ACPClogmsgTodata(msg,myname,BigBlindfirst=False,flagfoldseen=True):

    data={} #字典

    #print('in msg=',msg)
    m1=re.search("^STATE:(\d*):([^:]*):(.*):(.*):(.*)",msg.strip())
    #print('m1=',m1)
    
    hand_id=int(m1.group(1))
    actions=m1.group(2).strip()
    cards=m1.group(3)
    if BigBlindfirst:
        win_money=m1.group(4).split('|')
        playernames=m1.group(5).split('|')
        win_money.reverse()
        playernames.reverse()
    else:
        win_money=m1.group(4).split('|')
        playernames=m1.group(5).split('|')

    room_number=2
    position=playernames.index(myname)
    #print('position=',position)
    name=myname
    opposition=1-position
    opname=playernames[opposition]
    

    m2=re.search("([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",actions)
    #print('m2=',m2,m2.group(0))
    preflop_actions = m2.group(1)
    flop_actions = m2.group(2)
    turn_actions = m2.group(3)
    river_actions = m2.group(4)
    lastaction=''
    if actions:
        lastaction=actions[-1]
    if preflop_actions=='':
        flagFirstAction=True
    else:
        flagFirstAction=False

    #print("cards",cards)
    m3 = re.search("([^\|]*)\|([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",cards.strip())
    #print('m3=',m3,m3.group(0))
    if BigBlindfirst:
        hand_p1=m3.group(2)
        hand_p2=m3.group(1)
    else:
        hand_p1=m3.group(1)
        hand_p2=m3.group(2)
    flopcds=m3.group(3)
    turncds=m3.group(4)
    rivercds=m3.group(5)
    #print("flopcds=",flopcds)
    #print("turncds=",turncds)
    #print("rivercds=",rivercds)

    #位置和局序号
    data['hand_id']=hand_id

    #手牌
    if position==0 :
        data['private_card']=[hand_p1[:2],hand_p1[2:]]
    else:
        data['private_card']=[hand_p2[:2],hand_p2[2:]]

    #print('hand_p1=',hand_p1)
    #print('hand_p2=',hand_p2)
    if hand_p1 and hand_p2 :
        data['player_card']=[[hand_p1[:2],hand_p1[2:]],[hand_p2[:2],hand_p2[2:]]]
    else:
        if position==0 :
            data['player_card']=[[hand_p1[:2],hand_p1[2:]],[]]
        else:
            data['player_card']=[[],[hand_p2[:2],hand_p2[2:]]]
    
    ophand=data['player_card'][opposition]
    if ophand:
        ophandidx=HandtoIdx(cardint[ophand[0]],cardint[ophand[1]])
        data['ophandidx']=ophandidx

    if not flagfoldseen and lastaction=='f': #当对手手牌数据不可观测时，根据动作f来去掉
        data['player_card'][opposition]=[]


    #公共牌
    street=1
    actionstrings=[preflop_actions]
    data['public_card']=[]
    if flopcds :
        data['public_card']=[flopcds[:2],flopcds[2:4],flopcds[4:]]
        street=2
        actionstrings.append(flop_actions)
    if turncds :
        data['public_card'].append(turncds)
        street=3
        actionstrings.append(turn_actions)
    if rivercds :
        data['public_card'].append(rivercds)
        street=4
        actionstrings.append(river_actions)
    data['street']=street
    #print('actionstrings=',actionstrings)

    #动作和下注额
    #要特别注意：下面的代码是很多地方都是基于两人对局做了特殊处理的
    #A raise is valid if a) it raises by at least one chip, b) the player has sufficient money in their stack to
    #raise to the value, and c) the raise would put the player all-in (they have spent all their chips) or the
    #amount they are raising by is at least as large as the big blind and the raise-by amount for any other
    #raise in the round. 
    #注意acpc协议中r****实际是raiseto，就是全局加注到多少(而不是一个轮次的)
    #一次raise实际下注额是call上的+对手raise增加的额度+大盲注
    #那么raiseto的最小值为：原来已经下注的额度+前一次raise增加的额度+大盲注
    money_beted_now=[params_sb,params_bb] #两个人对局：位置0的下注额，位置1的下注额
    money_beted_rnd=[0,0,0,0] #四个轮次 
    actions=[]
    actions_all=[]
    tmp_lastAction=[] #最后一个动作，倒数第一个动作
    tmp_lastSecAction=[] #最后第二个动作，倒数第二个动作
    tmp_lastThiAction=[] #倒数第三个动作
    raisetimes=0
    raisemprev=0
    actpos=0    #各动作的所对应的玩家的位置
    for i  in range(len(actionstrings)):
        #tmp_lastRoundAction='' #当前轮的动作记录
        #tmp_lastMyRoundAction='' #当前轮的动作记录

        actions_remainder=actionstrings[i]
        actions_round=[]
        raisetimes=0 #一轮中raise的次数，各轮需要分别统计
        raisemprev=0 #一轮中一次raise增加的下注额度，各轮需要分别统计
        actsn=0      #一轮中动作的序号
        while actions_remainder != '':
            actsn+=1 #每一轮第一个动作序号为1，第二个动作序号为2，后续为3,4,5...
            if i==0: #preflop轮小盲先行
                actpos=1-actsn%2
            else: #flop/turn/river轮大盲先行
                actpos=actsn%2
            #print('actsn=',actsn)
            #print('actpos=',actpos)
            parsed_chunk = ''
            if actions_remainder.startswith("c"):
                #call 改成check
                if money_beted_now[actpos]==money_beted_now[1-actpos]:
                    actionstr="check"
                else:
                    actionstr="call"
                actions_round.append({"action":actionstr,"position": actpos})
                actions_all.append({"action":actionstr,"position": actpos})
                parsed_chunk = "c"
                money_beted_now[actpos]=money_beted_now[1-actpos]
            elif actions_remainder.startswith("r"):
                #print('actions_remainder=',actions_remainder)
                raise_amount = int(re.search("^r(\d*).*",actions_remainder).group(1))
                parsed_chunk = "r" + str(raise_amount)
                #print('raise_amount=',raise_amount)
                raisetimes+=1
                #raise的额度就是在call平基础上增加的额度
                #raisemprev=raise_amount-max(money_beted_now)
                #raise_amount就是raise后的下注额，即raiseto的额度
                money_beted_now[actpos]=raise_amount
                #注意acpc协议是全局的raiseto，而cisia则是当前轮的raiseto
                #acpc
                #actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                #actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
                #parsed_chunk = "r" + str(raise_amount)
                #cisia
                if i==0:
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
                else:
                    raise_amount=raise_amount-money_beted_rnd[i-1]
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
            elif actions_remainder.startswith("f"):
                actions_round.append({"action":"fold","position": actpos})
                actions_all.append({"action":"fold","position": actpos})
                parsed_chunk = "f"
            else:
                print("wrong action string")
            tmp_lastThiAction=tmp_lastSecAction
            tmp_lastSecAction=tmp_lastAction
            tmp_lastAction=[actions_round[-1]["action"],actpos]
            #tmp_lastMyRoundAction=tmp_lastRoundAction
            #tmp_lastRoundAction=actions_round[-1]["action"] #当前轮的动作记录
            
            actions_remainder = actions_remainder.replace(parsed_chunk,"",1)

        if money_beted_now[0]==money_beted_now[1]:
            money_beted_rnd[i]=money_beted_now[0]
        else:
            #print("not equal bet in round:",i)
            pass

        actions.append(actions_round)
    
    data['action_history']=actions


    #判断决策点信息
    #这里虽然可能轮次变化后最后一个动作可能是前一轮的，但很巧的是
    #下面的逻辑也没有问题，因为前一个动作若是r的画，那么轮次比如没有结束
    #所以面临的决策点是pcl是没有问题的。
    if position==0 :
        flagDecisionPt=1
    else:
        flagDecisionPt=0

    if tmp_lastAction:
        if tmp_lastAction[0][0]=="r":
            flagDecisionPt=1 #pcl point
        else:
            flagDecisionPt=0 #pck point
    if tmp_lastAction:
        if tmp_lastAction[1]==position:#当最后一个动作是自己的时就一定在轮次交界处
            #我方的动作必然是c或者f
            #因此对手的动作必然是call或者raise
            #若最后一个动作在第一轮，则对手必然是在pcl
            if street==2: # cc|flop，rc|flop
                flagOpDecisionPt=1 #pcl point,前一次对手面对的决策点类型
            else: # 比如：  rrc|turn
                if tmp_lastThiAction:
                    if tmp_lastThiAction[0][0]=='r':
                        flagOpDecisionPt=1
                    else: # 比如：crc|turn
                        flagOpDecisionPt=0 #pck point,前一次对手面对的决策点类型
                else:
                    flagOpDecisionPt=1
        else:
            if tmp_lastSecAction:
                if tmp_lastSecAction[0][0]=='r':
                    flagOpDecisionPt=1 
                else:
                    flagOpDecisionPt=0
            else:# 若倒数第二个动作不存在，且不是我做的倒数第一个动作，必然是对手做的，那么就是每局开始的时候
                flagOpDecisionPt=1
    else:#没有动作则无需考虑
        flagOpDecisionPt=-1

    data['LastAction']=tmp_lastAction
    #data['LastMyAction']=tmp_lastMyAction
    data['flagFirstAction']=flagFirstAction
    data['flagDecisionPt']=flagDecisionPt
    data['flagOpDecisionPt']=flagOpDecisionPt

    #玩家信息
    data['players']=[]
    for i in range(room_number):
        data['players'].append({"position":i,"money_bet":money_beted_now[i],"money_left":params_stack-money_beted_now[i],'total_money':params_stack})
    data['roundbet']=money_beted_rnd

    #legal_action
    legalact=[]
    #当前最小的加注额应等于大盲注+对手的加注额度(对手在平基础上增加的额度)
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]不相等时，必然还在一个轮次内
    # 那么对手的加注额必然是：abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]相等时，要么新一轮开始或者一轮前面对手c
    # 那么对手的加注额为0也等于abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 所以最小加注额如下是对的：
    raisetorangemin=abs(money_beted_now[actpos]-money_beted_now[1-actpos])+params_bb
    
    if money_beted_now[actpos]==money_beted_now[1-actpos]:#这里是典型的基于2人的考虑的处理
        legalact.append("check")
        #通常相等的时候raise是没有多次的
        legalact.append("raise")
    else:
        legalact.append("call")
        legalact.append("fold")
        if raisetimes<params_maxr:
            legalact.append("raise")
    data["legal_actions"]=legalact
    if "raise" in legalact:
        raisetorangemin+=max(money_beted_now)
        if raisetorangemin>params_stack:
            raisetorangemin=params_stack

    #注意acpc的服务器的raise的额度可以自动调整的，默认是raiseto，若这个值过小，则会自动调整调整到call后加上该值 
    #所以提示的额度最好使用加注至(即raiseto)，分两种，一种是而且是全局的"raise_to_range"，
    #另一种是像自动化所那样使用一轮的raiseto，使用"raise_range"。
    if "raise" in legalact:#raise的范围是一轮中的值
        if street ==1:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto
            data["raise_range"]=[raisetorangemin,params_stack]
        else:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto。因为从第二轮开始street=2，要减去第一轮的下注，第一轮在money_beted_rnd列表中位置是0。
            data["raise_range"]=[raisetorangemin-money_beted_rnd[street-2],params_stack-money_beted_rnd[street-2]]
    
    #info,action_position
    data['info']='state'
    if lastaction=='f' or (hand_p1 and hand_p2) :
        actpos=2 #为了给出action_position=-1，所以设置为2
        data["info"]="result"
    data['position']=position

    #print("actions_all=",actions_all,len(actions_all))
    #print("actions=",actions,len(actions))
    #print("street=",street)
    #if street==2: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==3: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==4: print("str=",street,actions[street-1],len(actions[street-1]))
    if len(actions_all)==0:#当全局没有任何动作时是小盲
        data["action_position"]=0
    elif street==2 and len(actions[1])==0:#当flop没有任何动作时是大盲
        data["action_position"]=1
    elif street==3 and len(actions[2])==0:#当turn没有任何动作时是大盲
        data["action_position"]=1
    elif street==4 and len(actions[3])==0:#当river全局没有任何动作时是大盲
        data["action_position"]=1
    else:#其它情况下，当前需要动作的玩家是最后一个动作的玩家的对手
        data["action_position"]=1-actpos 


    #一局的结果可以直接算出来
    '''
    win_money=[0,0]
    if hand_p1 and hand_p2:
        rk1=gethandrank(data['player_card'][0],data['public_card'])
        rk2=gethandrank(data['player_card'][1],data['public_card'])
        if rk1>rk2:
            win_money=[money_beted_now[1],-money_beted_now[1]]
        elif rk1==rk2:
            win_money=[0,0]
        else:
            win_money=[-money_beted_now[0],money_beted_now[0]]
    if lastaction=='f':
        win_money=[0,0]
        foldpos=actions_all[-1]["position"]
        win_money[foldpos]=-money_beted_now[foldpos]
        win_money[1-foldpos]=money_beted_now[foldpos]
    '''

    if data["info"]=="result":
        for i in range(room_number):
            data['players'][i]["win_money"]=win_money[i]
    
    data['players'][position]["name"]=name
    data['players'][1-position]["name"]=opname

    return data




# 记录全部局数的整个信息集历史完整
#每条记录的格式为：playername,streetflag,public_card,recOpLastbet,action,player_card
recAllgameHistory=[]   #用betto来表示的历史
recAllgameHistoryS=[]  #用check，call，r+number表示的历史
recAllgameHistoryF=[]  #F表示first 一个赢率只记录两个决策点各一次
recAllgameHistorySF=[] #一个赢率只记录两个决策点各一次
#------------------------------------------------------------
#获取动作信息集
#不要用call/raise这样的动作来描述，用bet来描述，就是投入的金额数来描述。
#记录到recAllgameHistory中的信息为：
#playername,streetflag,public_card,recOpLastbet,ourbet,player_card
#记录到recAllgameHistoryS中的信息为：
#playername,streetflag,public_card,recOpLastact,ouract,player_card
def get_actioninfosetall(data,nplayer):
    global recAllagentsname
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次

    #--计算每一轮的commit出来，方便加上raise值的转换
    commitchipres=[] #--记录每一轮结束的commit
    chipsnres=[]     #记录每个人的commit情况
    foldflagres=[]   #记录每个人的fold情况

    for i in range(nplayer):
        chipsnres.append(0)
        foldflagres.append(0)

    chipsnres[0]=50
    chipsnres[1]=100

    streetflag=0
    recordsn=[0,0,0,0] #--用于记录各个轮次玩家的行动数量
    recOpLastbet="none"
    recOpLastbetS="none"

    #recAllagentsstat[name]={"chand":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
    #   "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0} 
    cpfvplay={}
    cpfvraise={}
    cafraise={}
    cafcall={}
    cfhandpre={}

    cafcck={}
    cafcckck={}
    cafcckrs={}
    cafccl={}
    cafcclfd={}
    cafcclcl={}
    cafcclrs={}

    cpfccl={}
    cpfcclfd={}
    cpfcclrs={}
    cpfcclcl={}
    cpfcck={}
    cpfcckrs={}
    cpfcckck={}

    cpfall={}
    cpfallfd={}
    cpfallcl={}
    cafall={}
    cafallfd={}
    cafallcl={}

    for name in recAllagentsname:
        cpfvplay[name]=False
        cpfvraise[name]=False
        cafraise[name]=0
        cafcall[name]=0
        cfhandpre[name]=0

        cafcck[name]=0
        cafcckrs[name]=0
        cafcckck[name]=0
        cafccl[name]=0
        cafcclfd[name]=0
        cafcclcl[name]=0
        cafcclrs[name]=0

        cpfccl[name]=0
        cpfcclfd[name]=0
        cpfcclrs[name]=0
        cpfcclcl[name]=0
        cpfcck[name]=0
        cpfcckrs[name]=0
        cpfcckck[name]=0

        cpfall[name]=0
        cpfallfd[name]=0
        cpfallcl[name]=0
        cafall[name]=0
        cafallfd[name]=0
        cafallcl[name]=0

    for lstaction in data["action_history"]:#--#一局的所有动作列表

        #print('len(lstaction)',len(lstaction))
        tmp_preAct='none'
        flg_CKcaseCounted={}
        flg_CLcaseCounted={}
        flg_CKcaseidx={}  #标记每一轮玩家的面临决策点的第几次
        flg_CLcaseidx={}
        flg_CKcaseCur=False
        flg_CLcaseCur=False
        for name in recAllagentsname:
            flg_CKcaseCounted[name]=False
            flg_CLcaseCounted[name]=False
            flg_CKcaseidx[name]=0
            flg_CLcaseidx[name]=0

        if (len(lstaction)>0):
            #注意street=1表示preflop
            streetflag=streetflag+1  #根据data["action_history"]中的列表数量确定street即轮次
            tmp_i_cAct=0 #用于统计当前动作在这一轮中是第几个动作
            for dictactround in lstaction:#--#每一轮的动作字典
                playername=data["players"][dictactround['position']]['name']
                playerpos=data["players"][dictactround['position']]['position']
                tmp_i_cAct+=1
                if tmp_i_cAct==1:
                    if streetflag==1:
                        recOpLastbetS="bigbet"
                    else:
                        recOpLastbetS="none"

                #--start注意：这一段是针对两人做的处理
                #当处于大盲位且是第一轮时，若对手直接fold，那么我方就会失去决策的机会，这种情况要去除掉
                #处理时根据第一轮第一个动作进行判断，若第一个动作为fold，那么对手就失去了决策的机会要去掉
                #由于第一个动作必然是小盲注位(0号位)的行动，那么必然是1号位玩家失去机会
                if streetflag==1 and dictactround["action"]=="fold" and tmp_i_cAct==1:  
                        cfhandpre[data["players"][1]['name']]=1
                #--end注意----
                flg_outAction=False
                action=""
                recordsn[streetflag-1]=recordsn[streetflag-1]+1 #--记录这一轮的玩家的动作总数
                #--无论处于哪个位置，动作的意义的一样的
                #--check意味着不加钱，raise意味着本轮从0开始加到多少，call意味着把钱加到其他最大的值
                #--因为动作历史有顺序在里头，因此不会出现问题
                #--但是要记录fold的玩家，计算投注平衡时需要避开fold的玩家
                action1=dictactround["action"]
                if (dictactround["action"]== "call"):
                    if streetflag==1:
                        cpfvplay[playername]=True

                        #--start注意：这一段是针对两人做的处理
                        if (tmp_preAct=='none' or tmp_preAct[0]=='r'):
                            if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                                cpfall[playername]+=1
                                cpfallcl[playername]+=1
                            else:
                                if (not flg_CLcaseCounted[playername]):
                                    cpfccl[playername]+=1 #既可以call，fold，raise
                                    cpfcclcl[playername]+=1
                                    flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        #if cpfccl[playername]>1:
                        #   print('now call cpfccl=',cpfccl[playername])
                        #   anykey=input()
                        #--end注意----

                    else:
                        cafcall[playername]+=1

                        #--start注意：这一段是针对两人做的处理
                        if tmp_preAct[0]=='r':
                            if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                                cafall[playername]+=1
                                cafallcl[playername]+=1
                            else:
                                if (not flg_CLcaseCounted[playername]):
                                    cafccl[playername]+=1
                                    cafcclcl[playername]+=1
                                    flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                            '''
                            if playername=='TARD':
                                print('street=',streetflag,' preact=',tmp_preAct)
                                print('flg_CLcaseidx',playername,flg_CLcaseidx[playername])
                                print('flg_CLcaseCur',flg_CLcaseCur,flg_CKcaseCur)
                                print('call at data',data)
                                flg_outAction=True
                                anykey=input()
                            '''
                        #--end注意----

                    #--获取记录的chipsnres数组中的最大值
                    chipmax=max(chipsnres)
                    #--设置当前position为该最大值
                    chipsnres[dictactround["position"]]=chipmax
                    action="betto"+str(chipmax)
                elif (dictactround["action"]== "check"):
                    action="betto"+str(chipsnres[dictactround["position"]])

                    #--start注意：这一段是针对两人做的处理
                    if streetflag==1 and (tmp_preAct=='call' or tmp_preAct=='check'):
                        if (not flg_CKcaseCounted[playername]):
                            cpfcck[playername]+=1
                            cpfcckck[playername]+=1
                            flg_CKcaseCounted[playername]=True
                        flg_CKcaseidx[playername]+=1
                        flg_CKcaseCur=True
                        flg_CLcaseCur=False

                    if streetflag>1 and (tmp_preAct=='none' or tmp_preAct=='call' or tmp_preAct=='check'):
                        if (not flg_CKcaseCounted[playername]):
                            cafcck[playername]+=1
                            cafcckck[playername]+=1
                            flg_CKcaseCounted[playername]=True
                        flg_CKcaseidx[playername]+=1
                        flg_CKcaseCur=True
                        flg_CLcaseCur=False
                    #--end注意----

                elif (dictactround["action"]== "fold"):
                    #--将fold信息记录到foldflagres数组中
                    foldflagres[dictactround["position"]]=1
                    action="fold"
                    
                    #--start注意：这一段是针对两人做的处理
                    if streetflag==1 and (tmp_preAct=='none' or tmp_preAct[0]=='r') :
                        if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                            cpfall[playername]+=1
                            cpfallfd[playername]+=1
                        else:
                            if (not flg_CLcaseCounted[playername]):
                                cpfccl[playername]+=1    #既可以call，fold，raise
                                cpfcclfd[playername]+=1
                                flg_CLcaseCounted[playername]=True
                        flg_CLcaseidx[playername]+=1
                        flg_CKcaseCur=False
                        flg_CLcaseCur=True
                        #if cpfccl[playername]>1:
                        #   print('fold cpfccl=',cpfccl[playername])
                        #   anykey=input()
                    if streetflag>1 and tmp_preAct[0]=='r':
                        if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                            cafall[playername]+=1
                            cafallfd[playername]+=1
                        else:
                            if (not flg_CLcaseCounted[playername]):
                                cafccl[playername]+=1
                                cafcclfd[playername]+=1
                                flg_CLcaseCounted[playername]=True
                        flg_CLcaseidx[playername]+=1
                        flg_CKcaseCur=False
                        flg_CLcaseCur=True
                    #--end注意----

                else: #-- raise,从第二个字符开始的字符串转换成数字
                    if streetflag==1:
                        cpfvplay[playername]=True
                        cpfvraise[playername]=True

                        #--start注意：这一段是针对两人做的处理
                        if (tmp_preAct=='none' or tmp_preAct[0]=='r'):
                            if (not flg_CLcaseCounted[playername]):
                                cpfccl[playername]+=1 #既可以call，fold，raise
                                cpfcclrs[playername]+=1
                                flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        elif (tmp_preAct=='call' or tmp_preAct=='check'):
                            if (not flg_CKcaseCounted[playername]):
                                cpfcck[playername]+=1
                                cpfcckrs[playername]+=1
                                flg_CKcaseCounted[playername]=True
                            flg_CKcaseidx[playername]+=1
                            flg_CKcaseCur=True
                            flg_CLcaseCur=False
                        #--end注意----

                    else:
                        cafraise[playername]+=1

                        #--start注意：这一段是针对两人做的处理
                        if tmp_preAct=='none' or tmp_preAct=='call' or tmp_preAct=='check':
                            if (not flg_CKcaseCounted[playername]):
                                cafcck[playername]+=1
                                cafcckrs[playername]+=1
                                flg_CKcaseCounted[playername]=True
                            flg_CKcaseidx[playername]+=1
                            flg_CKcaseCur=True
                            flg_CLcaseCur=False
                        if tmp_preAct[0]=='r':
                            if (not flg_CLcaseCounted[playername]):
                                cafccl[playername]+=1
                                cafcclrs[playername]+=1
                                flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        #--end注意----

                    
                    raiseamount=int(dictactround["action"][1:])
                    betval=0
                    if(streetflag==1):
                        betval= raiseamount
                        pot=150
                    elif(streetflag==2):
                        betval=commitchipres[0]+raiseamount
                        pot=commitchipres[0]*nplayer
                    elif(streetflag==3):
                        betval=commitchipres[1]+raiseamount
                        pot=commitchipres[1]*nplayer
                    else:
                        betval=commitchipres[2]+raiseamount
                        pot=commitchipres[2]*nplayer
                    chipsnres[dictactround["position"]]=betval
                    action="betto"+str(betval)
                    
                    if raiseamount<=2*pot:
                        action1="r1"
                    else:
                        action1="r2"
                    if betval>=20000:
                        action1="allin"
                    
                    '''
                    if raiseamount<=pot:
                        action1="r1"
                    elif raiseamount<=2*pot:
                        action1="r2"
                    elif raiseamount<=4*pot:
                        action1="r3"
                    else:
                        action1="r4"
                    if betval>=20000:
                        action1="allin"
                    if betval<1000:
                        action1='r1'
                    elif betval<10000:
                        action1='r2'
                    elif betval<20000:
                        action1='r3'
                    '''

                reconeact=[playername,streetflag,data['public_card'],recOpLastbet,action,data['player_card'][playerpos],data['player_card'][1-playerpos]]
                reconeactS=[playername,streetflag,data['public_card'],recOpLastbetS,action1,data['player_card'][playerpos],data['player_card'][1-playerpos]]
                recAllgameHistory.append(reconeact)
                recAllgameHistoryS.append(reconeactS)

                #if flg_outAction:
                #   print('reconeactS',reconeactS)
                #   flg_outAction=False

                #通过分析表明当对手raise变成allin时，我方只能是call，无论决策是raise还是call，所以会造成模糊
                #因此我们去掉对手是allin动作的情况再进行统计。
                if flg_CLcaseidx[playername]==1 and flg_CLcaseCur and recOpLastbetS != "allin":
                    recAllgameHistoryF.append(reconeact)
                    recAllgameHistorySF.append(reconeactS)
                
                if flg_CKcaseidx[playername]==1 and flg_CKcaseCur:
                    recAllgameHistoryF.append(reconeact)
                    recAllgameHistorySF.append(reconeactS)
                
                recOpLastbet=action     #bet to 的形式 +fold
                recOpLastbetS=action1   #check，call，fold，分级的r1，r2，等
                tmp_preAct=dictactround["action"] #原始形式的记录


            #--判断非fold玩家是否投注已经平衡，如果平衡了说明当前轮已经结束了，否则是未结束的
            #--投注平衡的标记，也是当前轮是否结束的标志
            equiflag=True
            equichip=0
            for i in range(nplayer):
                if (foldflagres[i]!=1): #--随便找一个非fold的玩家来记录一个当前的投注额
                    equichip=chipsnres[i]
                    break

            mnotfold=0 #--统计一下没有fold玩家的数量
            for v in foldflagres:
                if(v==0):
                    mnotfold=mnotfold+1

            #--只有当当前轮的动作记录的数量不小于非fold玩家数量时，才有可能达到平衡
            if(recordsn[streetflag-1]>=mnotfold):
                for i in range(nplayer):
                    if (foldflagres[i]!=1 and chipsnres[i]!=equichip):
                        equiflag=False
                        break
            else:
                equiflag=False

            if(equiflag):
                commitchipres.append(equichip)
                #print("street",streetflag, "is finished")
            else:
                #print("street",streetflag, "is not finished")
                pass
    
    #统计到全局变量中去
    for name in recAllagentsname:
        if cpfvplay[name]:
            recAllagentsstat[name]["cpfvplay"]+=1
        if cpfvraise[name]:
            recAllagentsstat[name]["cpfvraise"]+=1
        recAllagentsstat[name]["cfhandpre"]+=cfhandpre[name]
        recAllagentsstat[name]["cafraise"]+=cafraise[name]
        recAllagentsstat[name]["cafcall"]+=cafcall[name]

        recAllagentsstat[name]["cafcck"]+=cafcck[name]
        recAllagentsstat[name]["cafcckrs"]+=cafcckrs[name]
        recAllagentsstat[name]["cafcckck"]+=cafcckck[name]
        recAllagentsstat[name]["cafccl"]+=cafccl[name]
        recAllagentsstat[name]["cafcclfd"]+=cafcclfd[name]
        recAllagentsstat[name]["cafcclcl"]+=cafcclcl[name]
        recAllagentsstat[name]["cafcclrs"]+=cafcclrs[name]

        recAllagentsstat[name]["cpfcck"]+=cpfcck[name]
        recAllagentsstat[name]["cpfcckrs"]+=cpfcckrs[name]
        recAllagentsstat[name]["cpfcckck"]+=cpfcckck[name]
        recAllagentsstat[name]["cpfccl"]+=cpfccl[name]
        recAllagentsstat[name]["cpfcclfd"]+=cpfcclfd[name]
        recAllagentsstat[name]["cpfcclcl"]+=cpfcclcl[name]
        recAllagentsstat[name]["cpfcclrs"]+=cpfcclrs[name]

        recAllagentsstat[name]["cpfall"]+=cpfall[name]
        recAllagentsstat[name]["cpfallcl"]+=cpfallcl[name]
        recAllagentsstat[name]["cpfallfd"]+=cpfallfd[name]
        recAllagentsstat[name]["cafall"]+=cafall[name]
        recAllagentsstat[name]["cafallcl"]+=cafallcl[name]
        recAllagentsstat[name]["cafallfd"]+=cafallfd[name]


    return None






# 玩家的特征统计并输出到文件中
def statPlayerFeature(opname='',nstt=0,ngames=0):

    global recAllagentsstat

    recAllagentsstat={}

    for name in recAllagentsname:
        #统计相关的次数，并计算特征量
        #{"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
        # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
        # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}
        recAllagentsstat[name]={"chand":0,"cfhandpre":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
        'cpfccl':0,'cpfcclfd':0,'cpfcclcl':0,'cpfcclrs':0,'cpfcck':0,'cpfcckck':0,'cpfcckrs':0,
        'cafccl':0,'cafcclfd':0,'cafcclcl':0,'cafcclrs':0,'cafcck':0,'cafcckck':0,'cafcckrs':0,
        'cpfall':0,'cpfallfd':0,'cpfallcl':0,'cafall':0,'cafallfd':0,'cafallcl':0,
        "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0,'rvpip':0.0,"rpfr":0.0,
        'br':0.0,'fr':0.0,'rr':0.0,'pfbr':0.0,'pffr':0.0,'pfrr':0.0}

    nplayer=len(recAllagentsname) 
    allbots="-".join(recAllagentsname)

    #用于记录整个过程中的特征参数
    vpipdata={}
    pfrdata={}
    afdata={}
    pfrdvpipdata={}
    pfbrdata={}
    pffrdata={}
    pfrrdata={}
    brdata={}
    frdata={}
    rrdata={}
    for name in recAllagentsname:
        vpipdata[name]=[]
        pfrdata[name]=[]
        afdata[name]=[]
        pfrdvpipdata[name]=[]
        pfbrdata[name]=[]
        pffrdata[name]=[]
        pfrrdata[name]=[]
        brdata[name]=[]
        frdata[name]=[]
        rrdata[name]=[]

    
    #处理每一局的结果数据
    if ngames==0:
        ngames=len(datausers)

    handid=0
    for data in datausers[nstt:nstt+ngames]:
        handid+=1

        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        get_actioninfosetall(data,nplayer)
        
        if 1:
            for name in recAllagentsname:
                recAllagentsstat[name]["chand"]=handid
                recAllagentsstat[name]["vpip"]=(recAllagentsstat[name]["cpfvplay"]/handid)
                recAllagentsstat[name]["pfr"]=(recAllagentsstat[name]["cpfvraise"]/handid)
                if handid-recAllagentsstat[name]["cfhandpre"] !=0:
                    recAllagentsstat[name]["rvpip"]=(recAllagentsstat[name]["cpfvplay"]/(handid-recAllagentsstat[name]["cfhandpre"]))
                    recAllagentsstat[name]["rpfr"]=(recAllagentsstat[name]["cpfvraise"]/(handid-recAllagentsstat[name]["cfhandpre"]))
                else:
                    recAllagentsstat[name]["rvpip"]=0
                    recAllagentsstat[name]["rpfr"]=0

                if recAllagentsstat[name]["cafcck"]!=0:
                    recAllagentsstat[name]["br"]=(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"])
                else:
                    recAllagentsstat[name]["br"]=0
                if recAllagentsstat[name]["cafccl"]!=0:
                    recAllagentsstat[name]["fr"]=(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"])
                    recAllagentsstat[name]["rr"]=(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"])
                else:
                    recAllagentsstat[name]["fr"]=0
                    recAllagentsstat[name]["rr"]=0

                if recAllagentsstat[name]["cpfcck"]!=0:
                    recAllagentsstat[name]['pfbr']=(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"])
                else:
                    recAllagentsstat[name]['pfbr']=0
                if recAllagentsstat[name]["cpfccl"]!=0:
                    recAllagentsstat[name]['pffr']=(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"])
                else:
                    recAllagentsstat[name]['pffr']=0
                if recAllagentsstat[name]["cpfccl"]!=0:
                    recAllagentsstat[name]['pfrr']=(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"])
                else:
                    recAllagentsstat[name]['pfrr']=0


                if recAllagentsstat[name]["cafcall"]!=0:
                    recAllagentsstat[name]["af"]=(recAllagentsstat[name]["cafraise"]/recAllagentsstat[name]["cafcall"]/1000)
                else:
                    recAllagentsstat[name]["af"]=0.1
                if recAllagentsstat[name]["vpip"]!=0:
                    recAllagentsstat[name]["pfrdvpip"]=(recAllagentsstat[name]["pfr"]/recAllagentsstat[name]["vpip"])
                else:
                    recAllagentsstat[name]["pfrdvpip"]=0.0
                
                vpipdata[name].append(recAllagentsstat[name]["vpip"])
                pfrdata[name].append(recAllagentsstat[name]["pfr"])
                afdata[name].append(recAllagentsstat[name]["af"])
                pfrdvpipdata[name].append(recAllagentsstat[name]["pfrdvpip"])
                pfbrdata[name].append(recAllagentsstat[name]["pfbr"])
                pffrdata[name].append(recAllagentsstat[name]["pffr"])
                pfrrdata[name].append(recAllagentsstat[name]["pfrr"])
                brdata[name].append(recAllagentsstat[name]["br"])
                frdata[name].append(recAllagentsstat[name]["fr"])
                rrdata[name].append(recAllagentsstat[name]["rr"])
                '''
                print('name\tpfbr\tpffr\tpfrr\tbf\tfr\trr')
                print('{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(name
                ,pfbrdata[name][-1],pffrdata[name][-1],pfrrdata[name][-1],
                brdata[name][-1],frdata[name][-1],rrdata[name][-1]))
                print('recAllagentsstat[name]=',recAllagentsstat[name])
                '''
                #anykey=input()

    print('recAllagentsstat=',recAllagentsstat)
    #anykey=input()

    #计算特征量
    handid=len(datausers)
    for name in recAllagentsname:
        recAllagentsstat[name]["chand"]=handid
        recAllagentsstat[name]["vpip"]=(recAllagentsstat[name]["cpfvplay"]/handid)
        recAllagentsstat[name]["pfr"]=(recAllagentsstat[name]["cpfvraise"]/handid)
        recAllagentsstat[name]["rvpip"]=(recAllagentsstat[name]["cpfvplay"]/(handid-recAllagentsstat[name]["cfhandpre"]))
        recAllagentsstat[name]["rpfr"]=(recAllagentsstat[name]["cpfvraise"]/(handid-recAllagentsstat[name]["cfhandpre"]))

        if recAllagentsstat[name]["cafcck"]!=0:
            recAllagentsstat[name]["br"]="{:.4f}".format(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"])
        else:
            recAllagentsstat[name]["br"]="{:.4f}".format(0)
        if recAllagentsstat[name]["cafccl"]!=0:
            recAllagentsstat[name]["fr"]="{:.4f}".format(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"])
            recAllagentsstat[name]["rr"]="{:.4f}".format(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"])
        else:
            recAllagentsstat[name]["fr"]="{:.4f}".format(0)
            recAllagentsstat[name]["rr"]="{:.4f}".format(0)

        if recAllagentsstat[name]["cpfcck"]!=0:
            recAllagentsstat[name]['pfbr']="{:.4f}".format(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"])
        else:
            recAllagentsstat[name]['pfbr']="{:.4f}".format(0)
        if recAllagentsstat[name]["cpfccl"]!=0:
            recAllagentsstat[name]['pffr']="{:.4f}".format(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"])
            recAllagentsstat[name]['pfrr']="{:.4f}".format(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"])
        else:
            recAllagentsstat[name]['pffr']="{:.4f}".format(0)
            recAllagentsstat[name]['pfrr']="{:.4f}".format(0)

        if recAllagentsstat[name]["cafcall"]!=0:
            recAllagentsstat[name]["af"]="{:.4f}".format(recAllagentsstat[name]["cafraise"]/recAllagentsstat[name]["cafcall"]/1000)
        else:
            recAllagentsstat[name]["af"]=1.0
        if recAllagentsstat[name]["vpip"]!=0:
            recAllagentsstat[name]["pfrdvpip"]="{:.4f}".format(recAllagentsstat[name]["pfr"]/recAllagentsstat[name]["vpip"])
        else:
            recAllagentsstat[name]["pfrdvpip"]=0.0
    print('recAllagentsstat=',recAllagentsstat)

    #将这些特征写入csv文件中
    featureslst=['chand','cfhandpre','cpfvplay','cpfvraise','cafraise','cafcall',
    'cpfccl','cpfcclfd','cpfcclcl','cpfcclrs','cpfcck','cpfcckck','cpfcckrs','cpfall','cpfallcl','cpfallfd',
    'cafccl','cafcclfd','cafcclcl','cafcclrs','cafcck','cafcckck','cafcckrs','cafall','cafallcl','cafallfd',
    'vpip','pfr','af','pfrdvpip','rvpip','rpfr','pfbr','pffr','pfrr','br','fr','rr']
    resfeaturelist=np.zeros((len(featureslst),len(recAllagentsname)))

    j=-1
    for name in recAllagentsname:
        j+=1
        for i in range(len(resfeaturelist)):
            resfeaturelist[i,j]=recAllagentsstat[name][featureslst[i]]
    featuresdata=[featureslst]+resfeaturelist.T.tolist()
    ftransposed = list()
    for i in range(len(featuresdata[0])):
        row = list()
        for sublist in featuresdata:
            row.append(sublist[i])
        ftransposed.append(row)
    featuresfinal=[["player"]+recAllagentsname]+ftransposed
    print('featuresfinal',featuresfinal)


    if opname !='':
        opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                 'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot']
        featureslst1=['vpip','pfr','af','pfrdvpip','rvpip','rpfr','pfbr','pffr','pfrr','br','fr','rr']
        resfeaturelist1=[]
        for i in range(len(featureslst1)):
            resfeaturelist1.append(float(recAllagentsstat[opname][featureslst1[i]]))
        resfeaturelist1.append(opnames.index(opname[:-9]))
        print('resfeaturelist1=',resfeaturelist1)


    '''
    #保存特征数据
    filenameft='rec'+"-"+allbots+"-"+str(handid)+'-featrues.csv'
    np.savetxt(filenameft,featuresfinal,delimiter=',',fmt ='% s') 

    for name in recAllagentsname:
        plt.figure() #绘图初始化
        plt.plot(pfbrdata[name],marker='x',markevery=123,label='PFBR') #
        plt.plot(pffrdata[name],marker=',',markevery=111,label='PFFR') #
        plt.plot(pfrrdata[name],marker='o',markevery=103,label='PFRR') #
        plt.plot(brdata[name],marker='v',markevery=127,label='BR') #
        plt.plot(frdata[name],marker='<',markevery=119,label='FR') #
        plt.plot(rrdata[name],marker='>',markevery=107,label='RR') #
        plt.title("Features for {}".format(name),fontsize=16) #
        plt.xlabel("games") #增加x轴说明
        plt.ylabel("value(%)") #增加y轴说明
        plt.legend(frameon=False,loc='upper right') #显示图例
        #plt.xlim(-5,100) #x轴绘制范围限制
        plt.ylim(-5,110) #x轴绘制范围限制
        plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+name+"-features.svg")
        plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+name+"-features.pdf")
    '''

    '''
    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(vpipdata[name],colorline[i],label=name) #
        i+=1
    plt.title("VPIP",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-vpip.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-vpip.pdf")

    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(pfrdata[name],colorline[i],label=name) #
        i+=1
    plt.title("PFR",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfr.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfr.pdf")

    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(afdata[name],colorline[i],label=name) #
        i+=1
    plt.title("AF",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-af.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-af.pdf")

    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(pfrdvpipdata[name],colorline[i],label=name) #
        i+=1
    plt.title("PFR/VPIP",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfrdvpip.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfrdvpip.pdf")
    
    plt.show()
    '''
    return resfeaturelist1


#从所有玩家的log文件准备数据
#特征数据+对手标签
def preparedata():
    filenames=[]
    #filenames.append(filename)
    #outfilename='Match-{}.{}-all.log'.format(playername,opname)

    playername='ASHE'
    opnames=['ElDonatoro','Feste','HITSZ','Hugh_iro','Hugh_tbr','Intermission','PokerBot5',
                'PokerCNN','PPPIMC','Rembrant6','RobotShark_iro','RobotShark_tbr','SimpleRule','Slumbot']
    
    #训练集数据准备
    nsamples=3000   #500局一个统计数据
    traindataset=[]
    for opname in opnames:
        for i in range(1,21):
            for j in range(2):
                filename="{}.{}.{}.{}.log".format(playername,opname,i,j)
                logfiledealing(filename,playername+'_2pn_2017')
                for k in range(int(3000/nsamples)):
                    sampledata=statPlayerFeature(opname+'_2pn_2017',k*nsamples,nsamples)
                    traindataset.append(sampledata)
        #print('anykey to conitnue')
        #anykey=input()
    
    trandatasetnp=np.array(traindataset)
    np.savetxt("data-train-all-opponents-nsamples-{}.csv".format(nsamples),trandatasetnp,delimiter =",",fmt ='%.6f')
    

    #测试集数据准备
    testdataset=[]
    for opname in opnames:
        for i in range(21,26):
            for j in range(2):
                filename="{}.{}.{}.{}.log".format(playername,opname,i,j)
                logfiledealing(filename,playername+'_2pn_2017')
                for k in range(int(3000/nsamples)):
                    sampledata=statPlayerFeature(opname+'_2pn_2017',k*nsamples,nsamples)
                    testdataset.append(sampledata)

        #print('anykey to conitnue')
        #anykey=input()
    testdatasetnp=np.array(testdataset)
    np.savetxt("data-test-all-opponents-nsamples-{}.csv".format(nsamples),testdatasetnp,delimiter =",",fmt ='%.6f')
    
    return None



class Recognet(torch.nn.Module): #conv-parallel
    def __init__(self,ninput,noutput):
        super(Recognet, self).__init__()
        

        self.classifier=torch.nn.Sequential(
            torch.nn.Linear(ninput, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, noutput)
            #torch.nn.Softmax()
        )


    def forward(self, x):
        output=self.classifier(x)
        return output



def testRecognet():
    model=Recognet(4,14)
    print('model',model)

    x=torch.rand(10,4)
    y=model(x)
    print('x',x)
    print('y',y)

    nsamples=500
    traindata=np.loadtxt("data-train-all-opponents-nsamples-{}.csv".format(nsamples),delimiter =",")
    
    x=torch.from_numpy(traindata[:10,:4]).float()
    ylabel=torch.from_numpy(traindata[:10,-1]).long()
    

    y=model(x)
    print('x',x)
    print('y',y,ylabel)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    error=loss_fn(y,ylabel)
    print('error=',error)

    return None



#不使用dataloader
def traintestmodel_old():

    nfeatures=6  # 取特征的数量
    nftindex=range(-6,0,1)
    noutput=14  #输出的类别总数，这是固定的，对于当前问题来说

    batch_size=10
    epoch_size=20000
    lean_rate=0.001
    flag_train=False #True #True #False

    ntrainsamples=3000
    ntestsamples=3000
    traindatasetfile="data-train-all-opponents-nsamples-{}.csv".format(ntrainsamples)
    testdatasetfile="data-test-all-opponents-nsamples-{}.csv".format(ntestsamples)
    
    nnmodelfile='recog-op-model.pkl'
    convergfile='recog-op-converge.dat'


    #创建输入和输出随机张量
    #读取测试数据
    try:
        if os.path.exists(traindatasetfile):
            traindata=np.loadtxt(traindatasetfile,delimiter =",")
            testdata=np.loadtxt(testdatasetfile,delimiter =",")

    except FileNotFoundError:
        print('warning: train data not loaded !')
    
    
    ##训练数据集
    dataxy=torch.from_numpy(traindata)
    x_train=dataxy[:,nftindex].float()
    y_train=dataxy[:,-1].long()

    N_data=len(y_train)
    batch_num=int(N_data//batch_size)
    print('x_train.size=',x_train,x_train.size())
    print('y_train.size=',y_train,y_train.size())
    

    ##测试数据集
    dataxy=torch.from_numpy(testdata)
    x_test =dataxy[:,nftindex].float()
    y_test =dataxy[:,-1].long()

    print('x_test.size=',x_test,x_test.size())
    print('y_test.size=',y_test,y_test.size())

    # 使用nn包将我们的模型定义为一系列的层。
    # nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
    # 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
    # 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
    model = Recognet(nfeatures,noutput)


    # nn包还包含常用的损失函数的定义；
    # 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
    # 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
    # 这是为了与前面我们手工计算损失的例子保持一致，
    # 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
    # 一般问题使用均方误差
    #loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
    #loss_fn = torch.nn.MSELoss(reduction='sum')
    #loss_fn = torch.nn.MSELoss(reduction='mean') #reduction='mean'
    #多分类问题使用交叉熵来作为损失函数
    #在交叉熵损失函数中输入是未归一化的预测及实际标签，会自动计算 Softmax 以及其对数。
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    # 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
    # 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法，如RMSProp等。
    # Adam构造函数的第一个参数告诉优化器应该更新哪些张量。
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lean_rate)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lean_rate)

    try:
        if os.path.exists(nnmodelfile):
            model.load_state_dict(torch.load(nnmodelfile))
    except FileNotFoundError:
        print('warning: parameter of the model was not loaded!')

    if flag_train:
        
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            model = model.cuda()
        

        #进入训练模式，以确保 dropout 和 batch normalization 等层处于训练模式
        model.train()

        epochs=epoch_size
        losshistory=np.zeros(epochs)
        for e in range(epochs):
            # 前向传播：通过向模型传入x计算预测的y。
            # 模块对象重载了__call__运算符，所以可以像函数那样调用它们。
            # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量。
            # 每次从大数据中抽取batch size的数据用于训练

            #当输入数据就是一个batch时，自动会用batch处理训练。
            #若输入数据是多个batch，那么在训练过程中加一个循环来逐个的batch处理
            lossofbatch=0.0
            for ibatch in range(batch_num):

                start_index = ibatch*batch_size
                end_index = (ibatch+1)*batch_size
                #这里为什么容易出问题，就是本来是有batchsize=1时，这么取后就没有了。
                x,y=x_train[start_index:end_index],y_train[start_index:end_index]  


                y_pred = model(x)
                #print('y_pred=',y_pred)
                #print('y=',y)

                # 计算并打印损失值。
                # 传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量。
                loss = loss_fn(y_pred,y)
                #print(e, loss.item())
                lossofbatch+=loss.item()

                # 反向传播之前清零梯度
                # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
                # model.zero_grad() #OR
                optimizer.zero_grad()

                # 反向传播：计算模型的损失对所有可学习参数的导数（梯度）。
                # 在内部，每个模块的参数存储在requires_grad=True的张量中，
                # 因此这个调用将计算模型中所有可学习参数的梯度。
                loss.backward()

                # 使用梯度下降更新权重。
                # 每个参数都是张量，所以我们可以像我们以前那样可以得到它的数值和梯度
                """
                with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
                """
                # 调用Optimizer的step函数使它所有参数更新
                optimizer.step()


            lossofepoch=lossofbatch/batch_num
            losshistory[e]=lossofepoch
            print("Epoch: {}/{}..loss:{} ".format(e+1, epochs, lossofepoch))

        if torch.cuda.is_available():
            x_train = x_train.cpu()
            y_train = y_train.cpu()
            model = model.cpu()
        
        
        #保存训练后的模块的参数
        torch.save(model.state_dict(),nnmodelfile)
        #print('model.parameters()=',list(model.parameters()))

        try:
            if os.path.exists(convergfile):
                dataconverge = np.loadtxt(convergfile, delimiter=',')
                totalhistory =dataconverge.tolist()+losshistory.tolist()
            else:
                totalhistory =losshistory
        except FileNotFoundError:
            print('warning: converge data not loaded!')
        
        np.savetxt(convergfile, totalhistory,  delimiter=',')

        #误差历史
        plt.figure()
        plt.plot(losshistory,label='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.legend()
        plt.title('Loss of training')


    if 1: #测试阶段


        #显示一下误差历史
        if os.path.exists(convergfile):
            dataconverge = np.loadtxt(convergfile, delimiter=',')
            plt.figure()
            plt.plot(dataconverge,label='loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.yscale('log')
            plt.legend()
            plt.title('Loss of training')


        #进入推理、评估模式，以确保设置 dropout 和 batch normalization 为评估。
        # 如果不这样做，有可能得到不一致的推断结果
        model.eval()

        #需要输出的测试集数量
        N_test=len(x_test)

        # 计算测试集输出
        with torch.no_grad():
            y_pred = torch.softmax(model.forward(x_test),dim=1)
        print('y_pred=',y_pred)

        y_prdt=torch.argmax(y_pred,dim=1)
        print('y_prdt=',y_prdt,y_prdt.size())
        print('y_test=',y_test,y_test.size())
        y_diff=np.abs((y_test-y_prdt).numpy())
        y_whdf=np.where(y_diff)[0] #返回非0的坐标，标签相同则应是0
        test_ern=len(y_whdf)  #非0的总数，也就是标签不相同的总数
        test_E=test_ern/len(x_test)
        print('where(y_diff)=',y_whdf,test_ern)
        
        errorclass=np.zeros(noutput)
        for idx in y_whdf:
            errorclass[y_test[idx]]+=1
        print('error_class=',errorclass)
        E_eachclass=np.array(errorclass)/len(x_test)*noutput

        print('E=',test_E,'ACC=',1-test_E)
        print('E_eachclass=',E_eachclass,'ACC_eachclass=',1-E_eachclass)

        filenameft='res-{}-trainsamples-{}-testsamples-{}-featrues.csv'.format(ntrainsamples,ntestsamples,nfeatures)
        resfinal=[['E_all',test_E]]
        resfinal.append(['ACC_all',1-test_E])
        for i in range(len(E_eachclass)):
            resfinal.append(['E_class{}'.format(i),E_eachclass[i]])
            resfinal.append(['ACC_class{}'.format(i),1-E_eachclass[i]])
        np.savetxt(filenameft,resfinal,delimiter=',',fmt ='% s') 


        plt.figure()
        ax = plt.axes()
        x1=torch.linspace(0, N_test , N_test)
        ax.plot(x1.numpy(), y_test[:N_test].numpy(),label='target')
        ax.plot(x1.numpy(), torch.argmax(y_pred[:N_test,:],dim=1).numpy(), '+',label='pred')
        ax.legend()
        plt.title('Prediction of Ngtr={} Ngtt={} Nft={}'.format(ntrainsamples,ntestsamples,nfeatures))
        plt.xlabel('Instance')
        plt.ylabel('Category')
        plt.savefig('fig-recog-op-pred-Ngtr-{}-Ngtt-{}-Nft-{}.pdf'.format(ntrainsamples,ntestsamples,nfeatures))

    plt.show()
    

    return None



#使用dataloader
def traintestmodel():

    nfeatures=9  # 取特征的数量
    #featureslst1=['vpip','pfr','af','pfrdvpip','rvpip','rpfr','pfbr','pffr','pfrr','br','fr','rr']
    nftindex=[2,4,5,6,7,8,9,10,11]
    noutput=14  #输出的类别总数，这是固定的，对于当前问题来说

    batch_size=100
    epoch_size=20000
    lean_rate=0.001
    flag_train=True #True #False

    ntrainsamples=1000
    ntestsamples=1000
    traindatasetfile="data-train-all-opponents-nsamples-{}.csv".format(ntrainsamples)
    testdatasetfile="data-test-all-opponents-nsamples-{}.csv".format(ntestsamples)
    
    nnmodelfile='recog-op-model-Ngtr-{}-Nft-{}.pkl'.format(ntrainsamples,nfeatures)
    convergfile='recog-op-converge.dat'


    #创建输入和输出随机张量
    #读取测试数据
    try:
        if os.path.exists(traindatasetfile):
            traindata=np.loadtxt(traindatasetfile,delimiter =",")
            testdata=np.loadtxt(testdatasetfile,delimiter =",")

    except FileNotFoundError:
        print('warning: train data not loaded !')
    
    
    ##训练数据集
    dataxy=torch.from_numpy(traindata)
    x_train=dataxy[:,nftindex].float()
    y_train=dataxy[:,-1].long()

    N_data=len(y_train)
    batch_num=int(N_data//batch_size)
    print('x_train.size=',x_train,x_train.size())
    print('y_train.size=',y_train,y_train.size())
    

    ##测试数据集
    dataxy=torch.from_numpy(testdata)
    x_test =dataxy[:,nftindex].float()
    y_test =dataxy[:,-1].long()

    print('x_test.size=',x_test,x_test.size())
    print('y_test.size=',y_test,y_test.size())

    # 使用nn包将我们的模型定义为一系列的层。
    # nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
    # 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
    # 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
    model = Recognet(nfeatures,noutput)


    # nn包还包含常用的损失函数的定义；
    # 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
    # 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
    # 这是为了与前面我们手工计算损失的例子保持一致，
    # 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
    # 一般问题使用均方误差
    #loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
    #loss_fn = torch.nn.MSELoss(reduction='sum')
    #loss_fn = torch.nn.MSELoss(reduction='mean') #reduction='mean'
    #多分类问题使用交叉熵来作为损失函数
    #在交叉熵损失函数中输入是未归一化的预测及实际标签，会自动计算 Softmax 以及其对数。
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')


    # 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
    # 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法，如RMSProp等。
    # Adam构造函数的第一个参数告诉优化器应该更新哪些张量。
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lean_rate)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lean_rate)

    try:
        if os.path.exists(nnmodelfile):
            model.load_state_dict(torch.load(nnmodelfile))
    except FileNotFoundError:
        print('warning: parameter of the model was not loaded!')

    if flag_train:
        
        if torch.cuda.is_available():
            print('CUDA is available')
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            model = model.cuda()

        traindataset=torch.utils.data.dataset.TensorDataset(x_train,y_train)
        trainloader=torch.utils.data.dataloader.DataLoader(dataset=traindataset,#pin_memory=True,
                                                       batch_size=batch_size,shuffle=True,num_workers=0)
        

        #进入训练模式，以确保 dropout 和 batch normalization 等层处于训练模式
        model.train()

        epochs=epoch_size
        losshistory=np.zeros(epochs)
        for e in range(epochs):
            # 前向传播：通过向模型传入x计算预测的y。
            # 模块对象重载了__call__运算符，所以可以像函数那样调用它们。
            # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量。
            # 每次从大数据中抽取batch size的数据用于训练

            #当输入数据就是一个batch时，自动会用batch处理训练。
            #若输入数据是多个batch，那么在训练过程中加一个循环来逐个的batch处理
            lossofbatch=0.0
            for ibatch,(batch_x,batch_y) in enumerate(trainloader):
                
                '''
                start_index = ibatch*batch_size
                end_index = (ibatch+1)*batch_size
                #这里为什么容易出问题，就是本来是有batchsize=1时，这么取后就没有了。
                x,y=x_train[start_index:end_index],y_train[start_index:end_index]  
                '''

                x,y=batch_x,batch_y

                y_pred = model(x)
                #print('y_pred=',y_pred)
                #print('y=',y)

                # 计算并打印损失值。
                # 传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量。
                loss = loss_fn(y_pred,y)
                #print(e, loss.item())
                lossofbatch+=loss.item()

                # 反向传播之前清零梯度
                # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
                # model.zero_grad() #OR
                optimizer.zero_grad()

                # 反向传播：计算模型的损失对所有可学习参数的导数（梯度）。
                # 在内部，每个模块的参数存储在requires_grad=True的张量中，
                # 因此这个调用将计算模型中所有可学习参数的梯度。
                loss.backward()

                # 使用梯度下降更新权重。
                # 每个参数都是张量，所以我们可以像我们以前那样可以得到它的数值和梯度
                """
                with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
                """
                # 调用Optimizer的step函数使它所有参数更新
                optimizer.step()


            lossofepoch=lossofbatch/batch_num
            losshistory[e]=lossofepoch
            print("Epoch: {}/{}.. batch loss:{} ".format(e+1, epochs, lossofepoch))

        if torch.cuda.is_available():
            x_train = x_train.cpu()
            y_train = y_train.cpu()
            model = model.cpu()
        
        
        #保存训练后的模块的参数
        torch.save(model.state_dict(),nnmodelfile)
        #print('model.parameters()=',list(model.parameters()))

        try:
            if os.path.exists(convergfile):
                dataconverge = np.loadtxt(convergfile, delimiter=',')
                totalhistory =dataconverge.tolist()+losshistory.tolist()
            else:
                totalhistory =losshistory
        except FileNotFoundError:
            print('warning: converge data not loaded!')
        
        np.savetxt(convergfile, totalhistory,  delimiter=',')

        #误差历史
        plt.figure()
        plt.plot(losshistory,label='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.legend()
        plt.title('Loss of training')


    if 1: #测试阶段


        #显示一下误差历史
        if os.path.exists(convergfile):
            dataconverge = np.loadtxt(convergfile, delimiter=',')
            plt.figure()
            plt.plot(dataconverge,label='loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.yscale('log')
            plt.legend()
            plt.title('Loss of training')


        #进入推理、评估模式，以确保设置 dropout 和 batch normalization 为评估。
        # 如果不这样做，有可能得到不一致的推断结果
        model.eval()

        #需要输出的测试集数量
        N_test=len(x_test)

        # 计算测试集输出
        with torch.no_grad():
            y_pred = torch.softmax(model.forward(x_test),dim=1)
        print('y_pred=',y_pred)

        y_prdt=torch.argmax(y_pred,dim=1)
        print('y_prdt=',y_prdt,y_prdt.size())
        print('y_test=',y_test,y_test.size())
        y_diff=np.abs((y_test-y_prdt).numpy())
        y_whdf=np.where(y_diff)[0] #返回非0的坐标，标签相同则应是0
        test_ern=len(y_whdf)  #非0的总数，也就是标签不相同的总数
        test_E=test_ern/len(x_test)
        print('where(y_diff)=',y_whdf,test_ern)
        
        errorclass=np.zeros(noutput)
        for idx in y_whdf:
            errorclass[y_test[idx]]+=1
        print('error_class=',errorclass)
        E_eachclass=np.array(errorclass)/len(x_test)*noutput

        print('E=',test_E,'ACC=',1-test_E)
        print('E_eachclass=',E_eachclass,'ACC_eachclass=',1-E_eachclass)

        filenameft='res-{}-trainsamples-{}-testsamples-{}-featrues.csv'.format(ntrainsamples,ntestsamples,nfeatures)
        resfinal=[['E_all',test_E]]
        resfinal.append(['ACC_all',1-test_E])
        for i in range(len(E_eachclass)):
            resfinal.append(['E_class{}'.format(i),E_eachclass[i]])
            resfinal.append(['ACC_class{}'.format(i),1-E_eachclass[i]])
        np.savetxt(filenameft,resfinal,delimiter=',',fmt ='% s') 


        plt.figure()
        ax = plt.axes()
        x1=torch.linspace(0, N_test , N_test)
        ax.plot(x1.numpy(), y_test[:N_test].numpy(),label='target')
        ax.plot(x1.numpy(), torch.argmax(y_pred[:N_test,:],dim=1).numpy(), '+',label='pred')
        ax.legend()
        plt.title('Prediction of Ngtr={} Ngtt={} Nft={}'.format(ntrainsamples,ntestsamples,nfeatures))
        plt.xlabel('Instance')
        plt.ylabel('Category')
        plt.savefig('fig-recog-op-pred-Ngtr-{}-Ngtt-{}-Nft-{}.pdf'.format(ntrainsamples,ntestsamples,nfeatures))

    plt.show()
    

    return None


def resultpostdeal():

    featurelst=[4,6,6,9,12]
    ftindexlst=[range(4),range(6),[6,7,8,9,10,11],[2,4,5,6,7,8,9,10,11],range(12)]
    nnmodellst=['','','A','','']
    ntrainslst=[500,750,1000,3000]
    ntestsplst=[500,750,1000,3000]
    #nnmodelfile='recog-op-model-Ngtr-{}-Nft-{}{}.pkl'.format(ntrainsamples,nfeatures)

    #测试第一个问题
    #训练特征和测试特征来自相同的局数，考察不同特征的对于影响
    #ntrainslst.reverse()
    for ntrainsamples in ntrainslst:
        ntestsamples=ntrainsamples
        print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
        print('{:^10} {:^10} {:^40} {:^10} {:^10}'.format('Ngtr','Ngtt','Nftidx','Error','ACC'))
        for i in range(len(featurelst)):
            nfeatures=featurelst[i]
            nftindex=ftindexlst[i]
            nnmodelfile='recog-op-model-Ngtr-{}-Nft-{}{}.pkl'.format(ntrainsamples,nfeatures,nnmodellst[i])
            Error,ACC=testmodel(nfeatures,nftindex,ntrainsamples,ntestsamples,nnmodelfile)
            print('{:^10} {:^10} {:^40} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,str(list(nftindex)),Error,ACC))
        

    #测试第二个问题
    #训练特征和测试特征来自不同的局数，考察不同特征对于识别率的影响
    for ntrainsamples in ntrainslst:
        for ntestsamples in ntestsplst:
            #if ntestsamples!=ntrainsamples:
            print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
            print('{:^10} {:^10} {:^40} {:^10} {:^10}'.format('Ngtr','Ngtt','Nftidx','Error','ACC'))
            for i in range(len(featurelst)):
                nfeatures=featurelst[i]
                nftindex=ftindexlst[i]
                nnmodelfile='recog-op-model-Ngtr-{}-Nft-{}{}.pkl'.format(ntrainsamples,nfeatures,nnmodellst[i])
                Error,ACC=testmodel(nfeatures,nftindex,ntrainsamples,ntestsamples,nnmodelfile)
                print('{:^10} {:^10} {:^40} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,str(list(nftindex)),Error,ACC))
        

    #测试第三个问题，区分不同特征输出到文件中
    #训练特征和测试特征来自不同的局数，考察不同特征对于识别率的影响
    for i in range(len(featurelst)):
        AllaccNeqsps=[ntestsplst]
        for ntrainsamples in ntrainslst:
            AllaccNeqspsrow=[]
            for ntestsamples in ntestsplst:
                #if ntestsamples!=ntrainsamples:
                print("\n Ngtr={}-Ngtt={}\n-----------------------".format(ntrainsamples,ntestsamples))
                print('{:^10} {:^10} {:^40} {:^10} {:^10}'.format('Ngtr','Ngtt','Nftidx','Error','ACC'))
            
                nfeatures=featurelst[i]
                nftindex=ftindexlst[i]
                nnmodelfile='recog-op-model-Ngtr-{}-Nft-{}{}.pkl'.format(ntrainsamples,nfeatures,nnmodellst[i])
                Error,ACC=testmodel(nfeatures,nftindex,ntrainsamples,ntestsamples,nnmodelfile)
                print('{:^10} {:^10} {:^40} {:^10f} {:^10f}'.format(ntrainsamples,ntestsamples,str(list(nftindex)),Error,ACC))
                AllaccNeqspsrow.append(ACC)
            AllaccNeqsps.append(AllaccNeqspsrow)
        AllaccNeqsps=torch.tensor(AllaccNeqsps).T.tolist()
        AllaccNeqsps=[[r'Ngtt\Ngtr']+ntrainslst]+AllaccNeqsps
        np.savetxt(f'ACC-ALL-Ngtr-Ngtt-Nft-{nfeatures}{nnmodellst[i]}.csv', AllaccNeqsps, fmt='%s', delimiter=',')

    
    return None


def testmodel(nfeatures,nftindex,ntrainsamples,ntestsamples,nnmodelfile):

    '''
    nfeatures=6  # 取特征的数量
    nftindex=range(-6,0,1)
    ntrainsamples=1000
    ntestsamples=1000
    nnmodelfile='recog-op-model-Ngtr-{}-Nft-{}.pkl'.format(ntrainsamples,nfeatures)
    '''

    testdatasetfile="data-test-all-opponents-nsamples-{}.csv".format(ntestsamples)
    noutput=14  #输出的类别总数，这是固定的，对于当前问题来说

    #创建输入和输出随机张量
    #读取测试数据
    try:
        if os.path.exists(testdatasetfile):
            testdata=np.loadtxt(testdatasetfile,delimiter =",")

    except FileNotFoundError:
        print('warning: train data not loaded !')
    

    ##测试数据集
    dataxy=torch.from_numpy(testdata)
    x_test =dataxy[:,nftindex].float()
    y_test =dataxy[:,-1].long()

    #print('x_test.size=',x_test,x_test.size())
    #print('y_test.size=',y_test,y_test.size())

    # 使用nn包将我们的模型定义为一系列的层。
    # nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
    # 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
    # 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
    model = Recognet(nfeatures,noutput)


    try:
        if os.path.exists(nnmodelfile):
            model.load_state_dict(torch.load(nnmodelfile))
    except FileNotFoundError:
        print('warning: parameter of the model was not loaded!')

    #进入推理、评估模式，以确保设置 dropout 和 batch normalization 为评估。
    # 如果不这样做，有可能得到不一致的推断结果
    model.eval()


    #需要输出的测试集数量
    N_test=len(x_test)

    # 计算测试集输出
    with torch.no_grad():
        y_pred = torch.softmax(model.forward(x_test),dim=1)
    #print('y_pred=',y_pred)

    y_prdt=torch.argmax(y_pred,dim=1)
    #print('y_prdt=',y_prdt,y_prdt.size())
    #print('y_test=',y_test,y_test.size())
    y_diff=np.abs((y_test-y_prdt).numpy())
    y_whdf=np.where(y_diff)[0] #返回非0的坐标，标签相同则应是0
    test_ern=len(y_whdf)  #非0的总数，也就是标签不相同的总数
    test_E=test_ern/len(x_test)
    test_ACC=1-test_E
    #print('where(y_diff)=',y_whdf,test_ern)
    
    errorclass=np.zeros(noutput)
    for idx in y_whdf:
        errorclass[y_test[idx]]+=1
    #print('error_class=',errorclass)
    E_eachclass=np.array(errorclass)/len(x_test)*noutput

    #print('E=',test_E,'ACC=',test_ACC)
    #print('E_eachclass=',E_eachclass,'ACC_eachclass=',1-E_eachclass)

    filenameft='res-recog-op-Ngtr-{}-Ngtt-{}-Nft-{}.csv'.format(ntrainsamples,ntestsamples,nfeatures)
    resfinal=[['E_all',test_E]]
    resfinal.append(['ACC_all',test_ACC])
    for i in range(len(E_eachclass)):
        resfinal.append(['E_class{}'.format(i),E_eachclass[i]])
        resfinal.append(['ACC_class{}'.format(i),1-E_eachclass[i]])
    np.savetxt(filenameft,resfinal,delimiter=',',fmt ='% s') 


    plt.figure()
    ax = plt.axes()
    x1=torch.linspace(0, N_test , N_test)
    ax.plot(x1.numpy(), y_test[:N_test].numpy(),label='target')
    ax.plot(x1.numpy(), torch.argmax(y_pred[:N_test,:],dim=1).numpy(), '+',label='pred')
    ax.legend()
    plt.title('Prediction of Ngtr={} Ngtt={} Nft={}'.format(ntrainsamples,ntestsamples,nfeatures))
    plt.xlabel('Instance')
    plt.ylabel('Category')
    plt.savefig('fig-recog-op-pred-Ngtr-{}-Ngtt-{}-Nft-{}.pdf'.format(ntrainsamples,ntestsamples,nfeatures))

    #plt.show()
    

    return test_E,test_ACC







if __name__ == "__main__":

    if 1:
        resultpostdeal()

    if 0:
        traintestmodel()

    if 0: #准备数据
        preparedata()
        pass

    if 0: #测试神经网络
        testRecognet()
        pass

    plt.show()






    

    

    
    



