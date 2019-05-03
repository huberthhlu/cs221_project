CS221 Project


Step 1.
  # -b <board_name>  -i <start> <end>
  #                       negative number denotes the last # page and '0' denotes the latest page
  python pttCrawler.py.py -b HatePolitics -i -3 0
Step 2. 
  python convert2sentences.py -b HatePolitics -i -3 0
  # step 1. and step 2. should share the same argument 
