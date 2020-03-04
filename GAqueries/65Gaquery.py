## 
# This code downloads all the daily data from Google Analytics API
#  Before running this, please make sure client_secrets.json is in the same folder
#
#
#
#
#
##
import argparse

from googleapiclient.discovery import build
import httplib2
from oauth2client import client
from oauth2client import file
from oauth2client import tools
import pandas as pd
import calendar as cl
import Credentials as cr
from datetime import datetime, timedelta,date
import datetime as dt
from collections import OrderedDict
import os.path

def authenticate_ga_api():
  """Initializes the analyticsreporting service object.

  Returns:
    analytics an authorized analyticsreporting service object.
  """
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      parents=[tools.argparser])
  flags = parser.parse_args([])

  # Set up a Flow object to be used if we need to authenticate.
  flow = client.flow_from_clientsecrets(
      cr.CLIENT_SECRETS_PATH, scope=cr.SCOPES,
      message=tools.message_if_missing(cr.CLIENT_SECRETS_PATH))

  '''Prepare credentials, and authorize HTTP object with them.
  If the credentials don't exist or are invalid run through the native client
  flow. The Storage object will ensure that if successful the good
  credentials will get written back to a file. '''
  storage = file.Storage('analyticsreporting.dat')
  credentials = storage.get()
  if credentials is None or credentials.invalid:
    credentials = tools.run_flow(flow, storage, flags)
  http = credentials.authorize(http=httplib2.Http())

  # Build the service object.
  analytics = build('analytics', 'v4', http=http, discoveryServiceUrl=cr.DISCOVERY_URI)
  return analytics

ga_auth = authenticate_ga_api()

def get_report(analytics, date, pageToken = None):
    if pageToken is None:
        return ga_auth.reports().batchGet(
                body={
                    'reportRequests': [
                    {
                        'viewId': VIEW_ID,
                        'dateRanges': date,
                        'metrics': [{'expression': 'ga:pageviews'}],
                        'dimensions': [{'name': 'ga:pagePath'},
                                       {'name': 'ga:date'}],
                       'dimensionFilterClauses': [{

                          'filters': [{

                              'dimension_name': 'ga:userAgeBracket',
                              'operator': 'EXACT',
                              'expressions': ["65+"]
                                    }]
                          }],
                       'samplingLevel': 'LARGE',
                       'pageSize' : 100000
                   }]
                }
            ).execute()
    else:
        return ga_auth.reports().batchGet(
                body={
                    'reportRequests': [
                    {
                       'viewId': VIEW_ID,
                        'dateRanges': date,
                        'metrics': [{'expression': 'ga:pageviews'}],
                        'dimensions': [{'name': 'ga:pagePath'},
                                       {'name': 'ga:date'}],
                       'dimensionFilterClauses': [{

                          'filters': [{

                              'dimension_name': 'ga:userAgeBracket',
                              'operator': 'EXACT',
                              'expressions': ["65+"]
                                    }]
                          }],
                       'samplingLevel': 'LARGE',
                       'pageToken' : pageToken,
                       'pageSize' : 100000
                   }]
                }
            ).execute()
    
def reportToList(report):
  list = []
  columnHeader = report.get('columnHeader', {})
  dimensionHeaders = columnHeader.get('dimensions', [])
  metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])
    
  for row in report.get('data', {}).get('rows', []):
      dict = {}
      dimensions = row.get('dimensions', [])
      dateRangeValues = row.get('metrics', [])

      for header, dimension in zip(dimensionHeaders, dimensions):
        dict[header] = dimension

      for i, values in enumerate(dateRangeValues):
        for metric, value in zip(metricHeaders, values.get('values')):
            #set int as int, float a float
            if ',' in value or '.' in value:
              dict[metric.get('name')] = float(value)
            else:
              dict[metric.get('name')] = int(value)
      list.append(dict)
  return list

def getGAData(startDate, endDate):
    analytics = authenticate_ga_api()
    list = []
    report = get_report(analytics, [{'startDate': startDate, 'endDate': endDate}]).get('reports', [])[0]
    report_data = report.get('data', {})
    if report_data.get('rowCount') is not None:
      print("Got: {} to {}. Row Count: {}".format(startDate, endDate, report_data.get('rowCount')))
      if report_data.get('samplesReadCounts', []) or report_data.get('samplingSpaceSizes', []):
          print("{} to {} contains sampled Data".format(startDate, endDate))
          return 'Sampled Data'
      if report_data.get('rowCount') > 900000:
          print("{} to {} exceeds pagination limit".format(startDate, endDate))
          return 'Exceeded Row Count'
      nextPageToken = report.get('nextPageToken')
      list = reportToList(report)
      while nextPageToken:
          print("\tIterating through pages. Token: {}".format(nextPageToken))
          report = get_report(analytics, [{'startDate': startDate, 'endDate': endDate}], nextPageToken).get('reports', [])[0]
          list = list + reportToList(report)
          nextPageToken = report.get('nextPageToken')
      print("Finalized Segment. Length: {}.".format(len(list)))
      return list
    

def getMonthData(year, month,startday,endday):
    lastDay = cl.monthrange(year, month)[1]
    indexDay = 1
    list = []
    if datetime.strptime(str(startday),'%Y-%m-%d').year == year and datetime.strptime(str(startday),'%Y-%m-%d').month == month:
        indexDay = datetime.strptime(str(startday),'%Y-%m-%d').day
    while indexDay <= lastDay:
      startDate = "{:%Y-%m-%d}".format(dt.datetime(year, month, indexDay))
      if startDate>endday:
        break;
      indexDay += 5
      if (indexDay > lastDay):
          indexDay = lastDay
      endDate = "{:%Y-%m-%d}".format(dt.datetime(year, month, indexDay))
      if endDate>endday:
          endDate = endday
      while True:
          response = getGAData(startDate, endDate)
          if type(response) != str and response is not None:
              list = list + response
              break;
          elif response is None:
              break;
          else:
              indexDay -= 1
              endDate = "{:%Y-%m-%d}".format(dt.datetime(year, month, indexDay))
      indexDay += 1
    if response is not None:
      df = pd.DataFrame(list)
      df.rename(columns={'ga:pageviews' : 'PageViews', 
                         'ga:pagePath' : 'PagePath',
                         'ga:date' : 'Date'}, inplace = True)
      df.reindex(['PagePath', 'Date','PageViews'], axis = 1)
      df['Date']=pd.to_datetime(df['Date'], format='%Y%m%d', errors = 'coerce').dt.date
      df = df.dropna()
      return df
    else:
      return None

def monthlist(dates):
    
  start, end = [datetime.strptime(_, '%Y-%m-%d') for _ in dates]
  L = OrderedDict(tuple(zip((int((start + timedelta(_)).strftime(r'%Y')), None),(int((start + timedelta(_)).strftime(r'%m')), None))) for _ in range((end - start).days)).keys()
  months = list(L)
  return months

def download_data(dates,date_start,date_end,filepath,data):
  months = monthlist(dates)
  for year, month in months:
    df = getMonthData(year, month,date_start,date_end)
    if df is not None:
      data = data.append(df,ignore_index=True)
  data.to_csv(filepath, encoding='utf-8', index=False, line_terminator='\n')


def checkfile(filepath, date_end):
  if os.path.isfile(filepath):
    data = pd.read_csv(filepath)
    last_date = data['Date'].max()
    latestdate = datetime.strptime(str(last_date),'%Y-%m-%d')
    datecompare = datetime.strptime(date_end,'%Y-%m-%d')
    if latestdate < datecompare:
      date_start = (latestdate + timedelta(days=1)).strftime('%Y-%m-%d')
      dates = [date_start,date_end]
      download_data(dates,date_start,date_end,filepath,data)
  else:
    date_start = '2014-11-01'
    dates = [date_start,date_end]
    data = pd.DataFrame()
    download_data(dates,date_start,date_end,filepath,data)

def main():
  VIEW_ID = cr.VIEW_ID
  filepath = './data/65GAData.csv'
  date_end = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

  checkfile(filepath,date_end)

if __name__ == '__main__':
  main()

