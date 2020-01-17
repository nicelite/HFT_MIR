from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as f
from time import strptime


def pre_processing(session, directory_path, write_path, stock_list):

    df = spark.read.csv(directory_path + '/*.csv', header=True)

    df = df.select(['t_price', 't_dt_neg', 't_dtm_neg', 't_isin', 't_q_exchanged'])\
        .withColumnRenamed('t_price', 'price')\
        .withColumnRenamed('t_dt_neg', 'date')\
        .withColumnRenamed('t_dtm_neg', 'time_micro')\
        .withColumnRenamed('t_isin', 'stock')\
        .withColumnRenamed('t_q_exchanged', 'volume')

    df = df.withColumn('date', f.concat(f.col('date'), f.lit('.'), f.col('time_micro')))
    df = df.withColumn('date', df['date'].cast('timestamp'))\
        .drop('time_micro')

    print('Storing on hdd ...')
    for stock_name in stock_list:
        print('Storing ' + stock_name[0] + ' on hdd ...')
        df_stock = df.filter(df['stock'] == stock_name[1])

        df_stock = df_stock.withColumn("id", f.monotonically_increasing_id())
        my_window = Window.partitionBy(f.window('date', '1 days')).orderBy('id')

        df_stock = df_stock.withColumn("prev_price", f.lag(df_stock['price']).over(my_window))
        df_stock = df_stock.withColumn("diff",
                                       f.when(f.isnull(df_stock['prev_price']), 0)
                                       .otherwise(f.log(df_stock['price']/df_stock['prev_price'])))
        df_stocks = df_stock.drop('pre_price')
        complete_write_path = write_path + '/' + stock_name[0] + '.csv'
        df_stock.toPandas()\
            .to_csv(complete_write_path, index=False)
        # df_stock.write.csv(complete_write_path, header=True)


def process_stock(session, stock_path, write_path, model):
    df = spark.read.csv(stock_path, header=True)
    df = df.filter('diff != 0.0')
    df = df.select('date', 'price', 'volume', 'diff')
    df = df.withColumnRenamed('diff', 'return')
    df = df.withColumn('datetime', f.concat(f.split(f.col('date'), '-').getItem(0),
                                            f.split(f.col('date'), '-').getItem(1),
                                            f.split(f.col('date'), '-| ').getItem(2),
                                            f.split(f.col('date'), ':| ').getItem(1),
                                            f.split(f.col('date'), ':| ').getItem(2),
                                            f.split(f.col('date'), ':|[.]').getItem(2),
                                            f.split(f.col('date'), '\w{2}[.]').getItem(1)))
    if model == 'pattern_recognition':
        # date as a numerical value
        df = df.drop('date')
        df = df.drop('volume')
        df = df.withColumnRenamed('datetime', 'date')

    df.toPandas()\
        .to_csv(write_path, index=False)


if __name__ == '__main__':
    conf = SparkConf().set('spark.driver.cores', '3')\
        .set('spark.executor.cores', '3')\
        .set('spark.driver.supervise', 'true')\
        .setAppName('pre_processing')\
        .setMaster('local')
    spark_context = SparkContext(conf=conf)

    spark = SparkSession(spark_context)

    liste_isin = [['Credit Agricole', 'FR0000045072'], ['Safran', 'FR0000073272'], ['Air Liquide', 'FR0000120073'],
                  ['Carrefour', 'FR0000120172'], ['Total', 'FR0000120271'], ["L'oreal", 'FR0000120321'],
                  ['Accor Hotels', 'FR0000120404'], ['Bouygues', 'FR0000120503'], ['Sanofi', 'FR0000120578'],
                  ['Axa', 'FR0000120628'], ['Danone', 'FR0000120644'], ['Pernod Ricard', 'FR0000120693'],
                  ['Lvmh', 'FR0000121014'], ['Michelin', 'FR0000121261'], ['Kering', 'FR0000121485'],
                  ['Peugeot', 'FR0000121501'], ['Essilor Intl', 'FR0000121667'], ['Klepierre', 'FR0000121964'],
                  ['Schneider Electric', 'FR0000121972'], ['Veolia Environ.', 'FR0000124141'],
                  ['Unibail-Rodamco', 'FR0000124711'], ['Saint Gobain', 'FR0000125007'], ['Cap Gemini', 'FR0000125338'],
                  ['Vinci', 'FR0000125486'], ['Vivendi', 'FR0000127771'], ['Alcatel-Lucent', 'FR0000130007'],
                  ['Publicis Groupe', 'FR0000130577'], ['Societe Generale', 'FR0000130809'],
                  ['Bnp Paribas', 'FR0000131104'], ['Technip', 'FR0000131708'], ['Renault', 'FR0000131906'],
                  ['Orange', 'FR0000133308'], ['Engie', 'FR0010208488'], ['Alstom', 'FR0010220475'],
                  ['Legrand SA', 'FR0010307819'], ['Airbus', 'NL0000235190']]

    # print('Pre processing Data ...')
    # pre_processing(session=spark,
    #                directory_path='D:/HFT',
    #                write_path='D:/HFT/stocks',
    #                stock_list=liste_isin)

    #liste_isin = [['Bnp Paribas', 'FR0000131104'], ['Sanofi', 'FR0000120578'],
    #              ['Total', 'FR0000120271']]
    # model = 'pattern_recognition'
    model = 'hft'
    print('Processing pre-processed Data ...')
    for stock in liste_isin:
        stock_path = 'D:/HFT/stocks/' + stock[0] + '.csv'
        process_stock(session=spark,
                      stock_path=stock_path,
                      write_path='D:/HFT/stocks_' + model + '/' + stock[0] + '.csv',
                      model=model)
