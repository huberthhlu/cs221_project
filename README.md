CS221 Project <br />
<br />
<br />
For PTT data collection
<br />
<br />
Step 1. <br />
   -b <board_name>  -i <start> <end> <br />
                         negative number denotes the last # page and '0' denotes the latest page  <br />
  python pttCrawler.py -b HatePolitics -i -3 0  <br />
<br />
<br />  
Step 2.  <br />
  python convert2sentences.py -b HatePolitics -i -3 0  <br />
  (step 1. and step 2. should share the same argument) <br />
<br />
<br />
For wiki data collection
<br />
<br />
python wikiCrawler.py -l 20  <br />
   (It means the number of collected articles from the category of "臺灣政治" on wikipedia is 20.)
