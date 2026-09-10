import requests
from typing import Optional, Dict, Any
from urllib.parse import quote

class NSE_Scrapper:
    nifty50 = "NIFTY 50"
    niftyBank = "NIFTY BANK"
    midCapNifty = "NIFTY MIDCAP 100"
    nifty100 = "NIFTY 100"
    niftyFinService = "NIFTY FIN SERVICE"
    niftyNext50 = "NIFTY NEXT 50"
    niftySmallCap = "NIFTY SMLCAP 100"
    niftyAuto = "NIFTY AUTO"
    niftyFMCG = "NIFTY FMCG"
    niftyIT = "NIFTY IT"
    niftyAlpha50 = "NIFTY ALPHA 50"
    
    
    url = "https://www.nseindia.com/api/NextApi/apiClient"
    indexTrackerApi = "/indexTrackerApi"
    quoteApi = "/GetQuoteApi"
    api = "/api"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Origin": "https://www.nseindia.com",
        "Connection": "keep-alive",
    }


    def _create_nse_session(self) -> requests.Session:
        """
        Create NSE session and set cookies.
        """
        session = requests.Session()
        session.headers.update(self.HEADERS)
        session.get("https://www.nseindia.com/", timeout=10)
        return session

    def _fetch_url(self,session: requests.Session,url: str,params: Optional[Dict[str, Any]] = None):
        """
        Fetch any URL using existing NSE session.

        Args:
            session : requests.Session
            url     : API URL
            params  : Optional query params

        Returns:
            Response JSON or text
        """
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        try:
            return response.json()
        except:
            return response.text


    def __init__(self):
        self.session = self._create_nse_session()

    
    def getIndexData(self):
        params = {
            "functionName": "getIndexData",
            "type": "All"
        }

        data = self._fetch_url(self.session, self.url, params=params)
        return data['data']
    
    def getMarketTurnOver(self):
        params = {
            "functionName": "getMarketTurnover",
        }
        data = self._fetch_url(self.session, self.url, params=params)
        return data['data']
    
    def getGiftNifty(self):
        params = {
            "functionName": "getGiftNifty"
        }
        data = self._fetch_url(self.session, self.url, params=params)
        return data['data']['giftNifty']
    
    def getUsdInr(self):
        params = {
            "functionName": "getGiftNifty"
        }
        data = self._fetch_url(self.session, self.url, params=params)
        return data['data']['usdInr']
    
    '''
    Index
    '''
    
    def getIndexAdvanceDecline(self,index):
        params = {
            "functionName": "getAdvanceDecline",
            "index": quote(index)
        }
        data = self._fetch_url(self.session, self.url + self.indexTrackerApi, params=params)
        return data['data']
        
    def getIndexConstituents(self,index):
        params = {
            "functionName": "getConstituents",
            "index": quote(index),
            "noofrecords":0
        }
        data = self._fetch_url(self.session, self.url + self.indexTrackerApi, params=params)
        return data['data']
    
    def getIndexHeatMap(self,index):
        params = {
            "functionName": "getIndicesHeatMap",
            "index": quote(index)
        }
        data = self._fetch_url(self.session, self.url + self.indexTrackerApi, params=params)
        return data['data']
    
    def getIndexCorporateAction( self,index,flag='CAC'):
        params = {
            "functionName": "getCorporateAction",
            "index": quote(index),
            "flag": quote(flag)
        }
        data = self._fetch_url(self.session, self.url + self.indexTrackerApi, params=params)
        return data['data']
    
    def getIndexCorporateAnnoucements(self,index,flag='CAN'):
        params = {
            "functionName": "getAnnouncementsIndices",
            "index": quote(index),
            "flag": quote(flag)
        }
        data = self._fetch_url(self.session, self.url + self.indexTrackerApi, params=params)
        return data['data']
    
    '''
    stock
    '''
    def _getStockMetaData(self,stock,series):
        params = {
            "functionName": "getSymbolData",
            "symbol": quote(stock.upper()),
            "series": quote(series),
            "marketType": "N"
            
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)['equityResponse']
        d = data[0]['metaData']
        d.update(data[0]['tradeInfo'])
        d.update(data[0]['priceInfo'])
        d.update(data[0]['secInfo'])    
        return d
    
    def getStockData(self,stock):
        params = {
            "functionName": "getMetaData",
            "symbol": quote(stock.upper()),
            
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        data.update(self._getStockMetaData(stock,data['activeSeries'][0]))
        return data
    
    def getStockCorporateAction(self,stock:str,noOfRecords:int=3):
        params = {
            "functionName": "getCorpAction",
            "symbol": quote(stock.upper()),
            "noOfRecords": quote(str(noOfRecords)),
            "marketApiType":"equities"
            
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data
    
    def getStockCorporateAnnoucements(self,stock:str,noOfRecords:int=3):
        params = {
            "functionName": "getCorporateAnnouncement",
            "symbol": quote(stock.upper()),
            "noOfRecords": quote(str(noOfRecords)),
            "marketApiType":"equities"
            
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data

    def getStockAnnualReports(self,stock:str,noOfRecords:int=3):
        params = {
            "functionName": "getCorpAnnualReport",
            "symbol": quote(stock.upper()),
            "noOfRecords": quote(str(noOfRecords)),
            "marketApiType":"equities"
            
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data        
    
    def getStockEventCalender(self,stock:str,noOfRecords:int=3):
        params = {
            "functionName": "getCorpEventCalender",
            "symbol": quote(stock.upper()),
            "noOfRecords": quote(str(noOfRecords)),
            "marketApiType":"equities"
            
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data   
    
    def getStockShareHoldingPattern(self,stock:str,noOfRecords:int=3):
        params = {
            "functionName": "getShareholdingPattern",
            "symbol": quote(stock.upper()),
            "noOfRecords": quote(str(noOfRecords)),
            "marketApiType":"equities"    
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data   
    
    def getStockFinancialStats(self,stock:str):
        params = {
            "functionName": "getFinancialStatus",
            "symbol": quote(stock.upper()),
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data   
    
    def getStockPeerComparisonData(self,stock:str,index=""):
        params = {
            "functionName": "getPeerComparisonData",
            "symbol": quote(stock.upper()),
            "type":"S",
            "quarter":"",
            "param":"industry",
            "index":index
        }
        data = self._fetch_url(self.session, self.url + self.quoteApi , params=params)
        return data  
    
    
    '''
    Derivatives
    '''
    
    def getMostActiveUnderlying(self):
        data = self._fetch_url(self.session, 'https://www.nseindia.com/api/live-analysis-most-active-underlying')['data']
        return data  
    
    def getLiveUnderlyingOI(self):
        data = self._fetch_url(self.session, 'https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings')['data']
        return data  
    
    def getOptionChainContractInfo(self,symbol:str):
        data = self._fetch_url(self.session, f'https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol.upper()}')
        return data  
    
    def getOptionChain(self,symbol:str,isIndex=True,expiry:str=''):
        '''
        expiry : dd-B-yyy
        '''
        expiry = self.getOptionChainContractInfo(symbol)['expiryDates'][0] if len(expiry) < 1 else expiry
        type = "Indices" if isIndex else "Equity"
        return self._fetch_url(self.session, f'https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol.upper()}&expiry={expiry}')
        
    
    '''Analysis'''
    def getAdvances(self):
       return self._fetch_url(self.session, 'https://www.nseindia.com/api/live-analysis-advance') 
    
    def getDeclines(self):
        return self._fetch_url(self.session, 'https://www.nseindia.com/api/live-analysis-decline') 
    
    def getUnchanged(self):
        return self._fetch_url(self.session, 'https://www.nseindia.com/api/live-analysis-unchanged') 
    
    def getStockList(self):
        return self._fetch_url(self.session, 'https://www.nseindia.com/api/live-analysis-stocksTraded') 
    
    def getMostActiveStocks(self,sortBy:str='value'):
        ''' sortby : value | volume '''
        return self._fetch_url(self.session, f'https://www.nseindia.com/api/live-analysis-most-active-securities?index={sortBy}') 