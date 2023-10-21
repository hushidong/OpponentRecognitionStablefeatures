
# A opponent recognition method based on stable features in Texas Hold'em 


## Data set 

DATA of MATCHES OF ACPC 2017 has been transferred to csv data with features.
(data from the matches with Player 'ASHE' against the other 14 players: 'ElDonatoro', 'Feste', 'HITSZ', 'Hugh\_iro', 'Hugh\_tbr', 'Intermission', 'PokerBot5', 'PokerCNN', 'PPPIMC', 'Rembrant6', 'RobotShark\_iro', 'RobotShark\_tbr', 'SimpleRule', 'Slumbot' )

each match corresponding to a record or several records, the format is like:

'vpip'|'pfr'|'af'|'pfrdvpip'|'rvpip'|'rpfr'|'pfbr'|'pffr'|'pfrr'|'br'|'fr'|'rr'|oppnent label
0.82|0.558|	0.2985|	0.6805|	0.878258|0.597644|0.9776|0.1282|0.5768|	0.9758|	0.3407|0.6484|1

there are total 12 features(id ranges from 0 to 11, the first 4 features are the traditional empirical features, the last 6 features are the stable features),  
and the opponent labels ranges from 0 to 13, corresponding to the above 14 opponents.


the features: 'pfbr','pffr','pfrr','br','fr','rr' are the features defined in the paper:
"Can Online Opponent Exploitation Earn More Than the Nash Equilibrium Computing Method in Incomplete Information Games with Large State Space?"
"Online Inference of Opponent's Hidden Information with Stable Features based Opponent Modeling in Texas Hold'em"


## Experiment

Each match has 3000 games. If we use the 3000 games to extract the features, then each match corresponding to a record. If we use the 500 games to extract the features, each match can generate 6 records.

Depending on the specific number of games (called Ng) using to extract the features, we build 4 type of recognizers with Ng=500,750,1000,3000, for each type, we use different features to train the 6 recognizers:

* recognizer 0 using  the first 4 features
* recognizer 1 using  the first 6 features
* recognizer 2 using  the last 6 features (i.e. the stable features)
* recognizer 3 using  the 9 features(features ids: 2,4,5,6,7,8,9,10,11)
* recognizer 4 using  the all 12 features

After training, we use the test data to test the recognition accuracy, 
the results are as follows:

### Results

Table. recognition accuracy with the first 4 features:

Ngtt\Ngtr|	500|	750|	1000|	3000
500|	0.777380943|	0.115476191|	0.158333331|	0.105952382
750|	0.12678571|	0.848214269|	0.194642857|	0.157142863
1000|	0.071428575|	0.309523821|	0.859523833|	0.142857149
3000|	0.071428575|	0.157142863|	0.200000003|	0.942857146


Table. recognition accuracy with the first 6 features:

  Ngtt\\Ngtr   500                   750                   1000                  3000
  ------------ --------------------- --------------------- --------------------- ---------------------
  500.0        0.7392857074737549    0.190476194024086     0.13809524476528168   0.0476190485060215
  750.0        0.19285714626312256   0.8196428418159485    0.19285714626312256   0.05000000074505806
  1000.0       0.14047619700431824   0.10238095372915268   0.8404762148857117    0.12857143580913544
  3000.0       0.1428571492433548    0.1428571492433548    0.1428571492433548    0.8999999761581421
  
  
Table. recognition accuracy with the last 6 features(i.e. stable features):

  Ngtt\\Ngtr   500                  750                  1000                 3000
  ------------ -------------------- -------------------- -------------------- --------------------
  500.0        0.8380952477455139   0.8702380657196045   0.8523809313774109   0.788095235824585
  750.0        0.8374999761581421   0.8821428418159485   0.8660714030265808   0.8089285492897034
  1000.0       0.8571428656578064   0.8928571343421936   0.8642857074737549   0.8404762148857117
  3000.0       0.8999999761581421   0.9357143044471741   0.9142857193946838   0.9642857313156128
  
  
Table. recognition accuray with the 9 features(i.e. features ids: 2,4,5,6,7,8,9,10,11):  
  
  Ngtt\\Ngtr   500                   750                  1000                 3000
  ------------ --------------------- -------------------- -------------------- --------------------
  500.0        0.9226190447807312    0.5071428418159485   0.6059523820877075   0.5916666388511658
  750.0        0.46785715222358704   0.8999999761581421   0.75                 0.6607142686843872
  1000.0       0.369047611951828     0.6619047522544861   0.8880952596664429   0.7428571581840515
  3000.0       0.1428571492433548    0.1428571492433548   0.5357142686843872   0.9714285731315613
  
Table. recognition accuray with the all 12 features:  
  
  Ngtt\\Ngtr   500                   750                   1000                  3000
  ------------ --------------------- --------------------- --------------------- --------------------
  500.0        0.8976190686225891    0.48452380299568176   0.6690475940704346    0.5071428418159485
  750.0        0.39642858505249023   0.8964285850524902    0.8125                0.5892857313156128
  1000.0       0.17619048058986664   0.5738095045089722    0.8999999761581421    0.6404761672019958
  3000.0       0.0714285746216774    0.1428571492433548    0.26428571343421936   0.9714285731315613
  
### Analysis

The recognizer with the stable features has the highest generalizing ability, becuase the the recognition accuracy of is obviously higher when different type of recognizer are applied to other data sets with different Ng (i.e. when Ngtr != Ngtt). Using more features including the 6 stable features, recognition accuray of the cases  Ngtr == Ngtt is slightly higher than using only the 6 stable features.
It can be concluded that the stable features is the more important than other features.


## Conclusion

This repo provides an opponent recoginiton methed based on stable features, experimental results show that the stable features can be used to recognize opponents and can archieve higher accuracy than the traditional empirical features.  

## history
2023-10-20


