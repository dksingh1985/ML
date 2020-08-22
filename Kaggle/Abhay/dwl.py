import urllib3


#Code 31 = Ashok Leyland 
url = 'https://www.business-standard.com/live-market/stocks/price-history-excel/coCode/31/fromYear//fromMonth//fromDay//toYear//toMonth//toDay//stockexchange/NSE'

http = urllib3.PoolManager()
response = http.request('GET',url)
#print(response.data )
body = str(response.data.decode('utf-8'))

f = open("historypri.html",'w')
f.write(body)#(response.read)
f.close()
