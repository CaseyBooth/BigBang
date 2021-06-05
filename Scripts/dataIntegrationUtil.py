# %load ./../functions/detect_peaks.py
"""Detect peaks in data based on their amplitude and other features."""

# from __future__ import division, print_function

import numpy as np
import os
import pandas as pd
import datetime
import scipy
from scipy import signal

__author__ = "Casey Booth"
__version__ = "1.0.0"
__license__ = ""

debug=True

#********   Standard Spraybit functions    *********


def LowPassFilt(dataIn,columnName):
    fs = 10
    cutoff = 0.5
    order = 2
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    dataIn[columnName] = signal.filtfilt(b, a, dataIn[columnName])

    return dataIn


#read in header rows with calval and df with data
def readCalVal(joinedfilename,garbageLines,old_board):
    # Old board config values to read header
    if old_board == True:
        header_rows = 2  # set rows in the header
        calval_index = 1  # set index value to find calval in df:calval
    # New board config values to read header
    else:
        header_rows = 5  # set rows in the header
        calval_index = 3  # set index value to find calval in df:calval

    # screen out garbage text at top of some data
    csvTest = 'x'
    while csvTest != 'Board':
        calvalScreen = pd.read_csv(joinedfilename, nrows=3, skiprows=garbageLines, header=None)
        csvTest = calvalScreen.at[0, 0]
        csvTest = csvTest[0:5]
        #             print('csvtest3= ',csvTest)
        if csvTest != 'Board':
            garbageLines += 1
        else:
            print('Found ', garbageLines, ' garbage lines')

    #  Read in calibration header df
    calval = pd.read_csv(joinedfilename, header=None, skiprows=garbageLines, nrows=header_rows,
                         names=['uuid', 'calval'])
    #         if calval.iloc[0]['uuid'] == "Enable data storage on your terminal APP. Send a '1' to proceed...":
    #             calval = pd.read_csv(joinedfilename, skiprows=1, header=None, nrows=header_rows, names=['uuid', 'calval'])
    #             header_rows = header_rows+1
    #             text_row=True

    #         print (calval)

    #   Check for delimiter error in calval df
    if pd.isnull(calval.iloc[calval_index, 1]):
        calval = pd.read_csv(joinedfilename, delimiter=':', skiprows=garbageLines,
                             header=None, nrows=header_rows, names=['uuid', 'calval'])

#         print (calval)

    if old_board:
        df = pd.read_csv(joinedfilename ,skiprows=garbageLines+header_rows+1,
                         names=['datetime', 'yaw', 'pitch', 'roll', 'ax', 'ay', 'az', 'temp', 'vhall3'])
#             print('old')
    else:
        df = pd.read_csv(joinedfilename ,skiprows=garbageLines+header_rows+1,
                         names=['datetime', 'yaw', 'pitch', 'roll', 'ax', 'ay', 'az', 'temp',
                                'vhall', 'vhall2', 'vhall3', 'vhall4'],error_bad_lines=True, warn_bad_lines=True)
        #             print('new')
    #     df = pd.read_csv(joinedfilename ,skiprows=3,names=['datetime', 'yaw', 'pitch', 'roll', 'ax', 'ay', 'az', 'temp', 'vhall3'])
    df.drop(df.head(2).index,inplace=True) #Drop the 2 first rows that have baseline signal values
    df.drop(df.tail(1).index,inplace=True) #Drop the last row - had some trash data in it
    df = df.fillna(method='ffill', limit=1)
    value = float(calval.iloc[calval_index]['calval'])

    return calval,df,calval_index,value

def readDataRate(df):

    #Convert datetime column to Pandas DateTime Format
    df['datetime'] =  pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M:%S.%f') #converting data type

    #capture data for summary report
    Start_date= df['datetime'].iloc[0]
    End_Date= df['datetime'].iloc[len(df)-1]
    Data_Points= len(df)

    #screen for only active study date data
    dft = df.copy() #copy to new to preserve data

    #Evaluate running data rate
    dft['gap'] = dft['datetime'].diff().dt.total_seconds()
    dft['localdatarate'] = 1 / dft['gap']
    dft.iloc[0, dft.columns.get_loc('localdatarate')] = 0
    dft.iloc[0, dft.columns.get_loc('gap')] = 0
    dft=dft.query('gap != 0' and 'localdatarate<1000')   #filter bad time data that will give infinite datarate
    # datarate=statistics.mean(dft.iloc[1:len(dft)]['localdatarate'])
    datarate=dft.iloc[1:len(dft)]['localdatarate'].mean()

    return dft,Start_date,End_Date,Data_Points,datarate

def dataCleanUp(df2,setpointFactor):
    
#     value = float(calval.iloc[calval_index]['calval'])
    vhallMax=df2.vhall3.max()
    setpoint = vhallMax*setpointFactor
#     print ('value = ', value)
    print ('vhallMax = ', vhallMax)
    print ('setpoint = ', setpoint)

    #This block of code is designed to eliminate a particularly irritating error where the first recorded
    # data is above the minimum peak threshold and causes an issue with finding the peaks feet

    #Find the first data point that is less than the setpoint and delete everything before that.
    firstpoint = df2[:].loc[df2.vhall3 < setpoint].head(1).index

#     if debug:
#         print('first =',firstpoint)  #to find files with  no data

    #Drop Data and redindex
    df2.drop(df2.index[:firstpoint[0]], inplace=True)
    df2 = df2.reset_index(drop=True)

    #This block of code is designed to eliminate another irritating error where the last recorded data is above
    # the minimum peak threshold and causes an issue with finding the peaks feet.
    # we will set the last 40 data pts to 0
    df2.iloc[-40:, df2.columns.get_loc('vhall3')] = 0
    return df2,setpoint,vhallMax
    
    

def eliminateMultiplePeaks(df2a,ind,setpoint):
    ind2 = ind.tolist()
    if debug:
        print('setpoint', setpoint)
        print('peaks to start', len(ind2))
    i = 0
    while i < len(ind2) - 1:
        #create list of vhall values from this peak index to next peak index
        findDupes = df2a['vhall3'].iloc[ind2[i]:ind2[i + 1]]
        #     print('i=',i)
        #     print(ind2)
        #determine how many points are below the setpoint
        belowSet = sum(k < setpoint for k in findDupes)
        #     print(belowSet)
        #if no data below setpoint, remove the next index and try again
        if belowSet == 0:
            ind2.pop(i + 1)
        else:
            i += 1
    #     if i>10:
    #         break
    if debug:
        print('peaks at end', len(ind2))

    # ind=np.array(ind2)
    ind = ind2
    return ind

def findStrokeStartEnd(df2d,i,ind,indValley,datarate,inddex,inddex2,width,width2):

    #find end of spray
    #create a list of all valley indexes greater than peak index-- grab first one
    indValley2=[item for item in indValley if item > i]
    tempindex = indValley2[0]
    vhallPeak=df2d['vhall3'].iloc[i]
    vhallFoot=df2d['vhall3'].iloc[tempindex]
    #define end of peak to ratio of the vhall signal at peak height
    findEnd=df2d[i:tempindex].loc[df2d.vhall3>((vhallPeak)*.5)].tail(1).index.values.tolist()
    tempindex = findEnd[0]
    inddex.append(findEnd[0])
#             inddex.append(indValley2[0])
#             tempindex = indValley2[0]

    #find begin of spray
    #create a list of all valley indexes less than peak index-- grab last one
    indValley2=[item for item in indValley if item < i]
    tempindex2=indValley2[-1]
    vhallStart=df2d['vhall3'].iloc[tempindex2]
    #define start of peak to ratio of the vhall signal at peak height
    findStart=df2d[tempindex2:i].loc[df2d.vhall3>((vhallPeak)*.5)].head(1).index.values.tolist()
#             findStart=df2d[tempindex2:i].loc[df2d.vhall3>setpoint].head(1).index.values.tolist()
    tempindex2=findStart[0]
    inddex2.append(findStart[0])

    #calculate width of peak
    print(i, end=', ') #Use this to find an issue
    totalDur=(tempindex - tempindex2) / datarate
    width.append(totalDur) #Total Duration
    width2.append(((i - tempindex2) / datarate)) #Functional Duration

    return tempindex,tempindex2,inddex,inddex2,width,width2,totalDur,vhallPeak

def dfOfEachStroke(df2d,tempindex,tempindex2,dferror):

    indexStartOfUse1 = tempindex2
    indexEndOfUse1 = tempindex
#             print('peak check found')
#             print(i,indexStartOfUse1,indexEndOfUse1)
    # refomat indexes to integers
    indexStartOfUse = int(indexStartOfUse1)
    indexEndOfUse = int(indexEndOfUse1)
#             print('peak check2 found')
    mask = (df2d['index'] >= indexStartOfUse) & (df2d['index'] <= indexEndOfUse)
    dfFullStroke= df2d.loc[mask]
#             print (dferror1.head())

    #concat this to dferror for a full list of all data points part of a spray
    dferror = pd.concat([dferror,dfFullStroke],sort=False)
#             print('peak errors found')
#             print('found errors')
    return dferror,dfFullStroke



def FR_DefineSprayDirection(df2a):

    conditions = [( df2a['pitch'] < 0 ) & (df2a['pitch'] > -20), 
                  ( df2a['pitch'] <= -20 ) & (df2a['pitch'] > -30),
                  ( df2a['pitch'] <= -30 ) & (df2a['pitch'] > -45),
                  ( df2a['pitch'] <= -45 ) & (df2a['pitch'] > -60),
                  ( df2a['pitch'] <= -60 ) & (df2a['pitch'] > -90),
                  ( (df2a['pitch']<10 )|(df2a['pitch']>-10 ) ) &( (df2a['roll']>170)|(df2a['roll']<-170)),
#                       ( (df2a['pitch']<10 )|(df2a['pitch']>-10 ) ) &( (df2a['roll']>-170)|(df2a['roll']<170)),
                  (df2a['pitch'] >0)&(df2a['pitch']<20),
                  (df2a['pitch'] >=20 )&(df2a['pitch']<45),
                  (df2a['pitch'] >=45)&(df2a['pitch']<90),       ]
    choices = [-20, -30,-45,-60,-90,-180,20,45,90]

    df2a['spraydirectionscore'] = np.select(conditions, choices)
    return df2a


def FR_DefineSprayDirectionV2(pitch,roll):

    #determine if pitching up or down
    if (pitch < 0):
        directionSign = -1
    else:
        directionSign = 1    

    #  use sum of squares to combine pitch and roll state into single vector
    #  to define the spray direction        
    directionVector = np.sqrt(pitch**2 + roll**2)

    #for vectors 0-120, we can handle the direction normally,
    #  as pitch is near horizontal the roll can greatly effect the vector value,
    #  we will use a bucket from 60-120 to capture near horizontal directions
    #  then as our direction vector goes >120 we need to calculate direction
    #  based the degrees pitch past 90
    if directionVector<120:
        sprayDirectionRaw=directionVector
    else:
        sprayDirectionRaw=90+(90-abs(pitch))

    conditions = [( sprayDirectionRaw >= 0 ) & (sprayDirectionRaw < 30), 
              (sprayDirectionRaw >= 30 ) & (sprayDirectionRaw < 45),
              (sprayDirectionRaw >= 45 ) & (sprayDirectionRaw < 60),
              (sprayDirectionRaw >= 60 ) & (sprayDirectionRaw < 120),
              (sprayDirectionRaw >= 120 ) & (sprayDirectionRaw < 150),
              (sprayDirectionRaw >= 150 ) & (sprayDirectionRaw<= 180),]
    choices = [15,45,60,90,135,180]

    #determine bucket the direction falls and apply direction value for up/down
    vector_direction = np.select(conditions,choices)*directionSign

    return vector_direction


                               




def defineSweepBehavior(dfMove,sweep,degPerSec,totalDur):

    #find degrees of movement for each data point
    #disregard direction positive or negative
    dfMove['sweep']=abs(dfMove['yaw'].diff())
#             print('complete sweep1')
    #filter out large movement of >20 degrees in 1 frame
    #which will be when the compass passes through 359 to 0 degrees or 0 to 359
    mask = (dfMove['sweep'] <= 20)
    dfMove=dfMove.loc[mask]

#             print('complete sweep2')
    #calculate total movement and speed
    degreesMov=dfMove['sweep'].sum()
    sweep.append(degreesMov)
    degPerSec.append(degreesMov/totalDur)

    return sweep, degPerSec

def defineShakeBehavior(df2c,i,dataStartMin,shakeList,shakeMax,accelThresh,datarate):

    #identify time window
    peakTime = pd.to_datetime(df2c['datetime'].iloc[i])
#             print('peak time =', peakTime)
    dataStart=peakTime - datetime.timedelta(minutes=dataStartMin)
#             print('shake Start =',dataStart)
    # filter by time prior to peak:
    mask = (df2c['datetime'] > dataStart) & (df2c['datetime'] <= peakTime)
    dfShake2 = df2c.loc[mask].copy()
#             print('shake time filtered')

    #create vector sum of accel data minus 1g
    dfShake2['combo']=(np.sqrt(dfShake2['ax']**2+dfShake2['ay']**2+dfShake2['az']**2))-1000

    #identify indexes with accel above 1g
    mask = (dfShake2['combo'] > accelThresh )
    dfShake3 = dfShake2.loc[mask].copy()
#             print('shake combo filtered')
    #count of indexes above 1g
    shakeRating = int(len(dfShake3)/datarate*1000) #ms of accel data over limit
    shakeMax1 = dfShake2['combo'].max()
#     print('count over 2g ',shakeRating)

    shakeList.append(shakeRating)
    shakeMax.append(int(shakeMax1)/1000)
#             print(shakeMax)
#             print('peak shake found')

    return shakeList,shakeMax



def defineMovementTime(dfMove,i,useTimeMin,datarate,movementDurationSecs):
    
    #get timestamp of the peak of spray
    peakTime = pd.to_datetime(dfMove['datetime'].iloc[i])
    #get timestamp prior to spray to begin looking for data to assess motion
    #   based on useTimeMin parameter 
    dataStartUse=peakTime - datetime.timedelta(minutes=useTimeMin)
    #filter by time prior to peak:
    #   Create df of data from x minutes before the spray, to the spray event
    mask = (dfMove['datetime'] > dataStartUse) & (dfMove['datetime'] <= peakTime)
    dfMove2 = dfMove.loc[mask].copy()
    #the number of data points in the df divided by data rate is seconds of motion prior to spray
    movementDurationSecs.append(len(dfMove2)/datarate)

    return movementDurationSecs
            






#    ***********   Smart Label Functions      *********


def createVectorSumMotion(dfVector):

    dfVector['MotionVector']=(np.sqrt(dfVector['ax']**2+dfVector['ay']**2+dfVector['az']**2))-1000

    return dfVector


def getMaxMotion(dfIn,startIndex,endIndex):

    dfIn=dfIn.loc[startIndex:endIndex]
    # print(startIndex,endIndex)
    # print(dfIn['MotionVector'])
    maxMotion=dfIn['MotionVector'].max()

    return maxMotion

def getUsageDetails(dfIn,idleTime):

#     df3 = dfIn.iloc[:, 0:1].copy()
    # df3 = dfIn.loc[:, 0:1].copy()
    df3 = dfIn.iloc[:, 0:1]
    df3 = df3.diff()  # Subtract each row from previous row to give time diff between samples
    df3['datetime'] = df3['datetime'].dt.total_seconds()  # Convert timediff to total seconds
    df3 = df3[~(df3['datetime'] < idleTime)]  # idle for 60 sec = new use
    #             print(df3)
    df3list = df3.index

    #  find end of use = 1 index before next wake boundary in df3
    #  use this to find the time at end of use to calculate duration
    #  also create df of all data of use get max motion value of the use for screening
    i = 1
    endOfUse = []
    maxMotion = []
    df3a = pd.DataFrame()
    for startUse in df3list:
        #                 print('startUse= ',startUse)
        #                 print('enduse = ',df3list[i])
        try:
            endOfUse.append(dfIn.at[df3list[i] - 1, 'datetime'])
            maxMotion.append(getMaxMotion(dfIn, startUse, df3list[i] - 1))
        except:
            endOfUse.append(dfIn.iloc[-1]['datetime'])
            maxMotion.append(getMaxMotion(dfIn, startUse, dfIn.index[-1]))
        i += 1
    #             print(endOfUse)
    print('df4 complete1')
    df3a['index'] = df3.index.tolist()
    df3a['EndTime'] = endOfUse
    df3a['maxMotion'] = maxMotion
    df3a.set_index('index', inplace=True)
    #             print(df3a)
    dfIn['EndTime'] = df3a['EndTime']
    dfIn['maxMotion'] = df3a['maxMotion']
    #             print (dfIn)

    # ----------------------------------------
    # Add column to dataframe for WAKE boundary

    #             dfIn['wakeboundary'] = df3.datetime
    #             print('wake bound')
    #             print(dfIn.head())

    # Add column to dataframe for DAY boundary
    print('df4 complete1.5')
    # df4 = dfIn.copy()
    df4 = dfIn.loc[:,:]
    # df4 = pd.DataFrame()
    print('df4 complete2')
    df4['dayofweek'] = df4['datetime'].dt.dayofweek
    df4['date'] = df4['datetime'].dt.date
    #             df4.drop_duplicates(['date'],keep='first', inplace=True)
    # df4list = df4.index
    print('df4 complete2.3')
    df4list = df4.index.tolist()
    print('df4 complete2.5')

    dfIn['dayboundary'] = df4.date
    dfIn.dropna(subset=['EndTime'], how='all', inplace=True)
    dfIn.drop(dfIn.columns[3:17], axis=1, inplace=True)
    print('df4 complete3')
#     dfIn=dfIn.drop(dfIn.columns[3:17], axis=1)
    dfIn['duration'] = dfIn['EndTime'] - dfIn['datetime']
    dfIn['duration'] = dfIn['duration'].dt.total_seconds()

    return dfIn


def formatUserName(userNameFile):
    dfUserNames = pd.read_csv(userNameFile, header=0)
    dfUserNames.drop(dfUserNames.columns[8:], axis=1, inplace=True)
    dfUserNames['Full Name']=dfUserNames['First Name']+ ' ' + dfUserNames['Last Initial']
    cols=['Visit 2 Code', 'RN#']
    dfUserNames['Month2 Code']= dfUserNames[cols].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
    dfReturn=dfUserNames[['Month2 Code','Full Name']].set_index('Month2 Code')
    # dfReturn.set_index('Month2 Code')

    print(dfUserNames.head())
    print(dfReturn.head())

    return dfReturn
