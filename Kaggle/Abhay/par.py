from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
     #Initializing lists
   lsStartTags = list()
   lsEndTags = list()
   lsStartEndTags = list()
   lsComments = list()

   #HTML Parser Methods
   def handle_starttag(self, startTag, attrs):
       self.lsStartTags.append(startTag)

   def handle_endtag(self, endTag):
       self.lsEndTags.append(endTag)

   def handle_startendtag(self,startendTag, attrs):
       self.lsStartEndTags.append(startendTag)

   def handle_comment(self,data):
       self.lsComments.append(data)

#creating an object of the overridden class
parser = MyHTMLParser()

#Opening NYTimes site using urllib2
#html_page = html_page = urllib2.urlopen("https://www.nytimes.com/")
html_page = open("historypri.html","r")

#Feeding the content
parser.feed(str(html_page.read()))

#printing the extracted values
print("Start tags", parser.lsStartTags)
print("End tags", parser.lsEndTags)
print("End tags ", parser.lsStartEndTags)
print("Comments", parser.lsComments)
