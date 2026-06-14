from utils import Sql, train_val_test_split
import numpy as np

sql_user='user-1'
sql_password='password'
sql_db='db_1'

if __name__ == '__main__':

    sql=Sql(sql_user, sql_password, sql_db)

    abalone_df = sql.query('SELECT * FROM abalone;')

    print(abalone_df)