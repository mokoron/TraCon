import time
import psycopg2
import torch


class PostgreDB:

    def __init__(self, options):
        self._options = options

        with psycopg2.connect(host=options.dbHost,
                                    user=options.dbUser,
                                    password=options.dbPassword,
                                    database=options.dbName) as conn:

            # create schema if not exists
            with conn.cursor() as cur:
                query = "CREATE SCHEMA IF NOT EXISTS " + options.dbSchema + ";"
                cur.execute(query)

    def get_connection(self):
        # set default schema
        return psycopg2.connect(host=self._options.dbHost,
                                    user=self._options.dbUser,
                                    password=self._options.dbPassword,
                                    database=self._options.dbName,
                                    options=f'-c search_path={self._options.dbSchema}')


def table_exists(cur, table, schema):
    query = "SELECT EXISTS ( "+\
            " SELECT FROM information_schema.tables " +\
            " WHERE  table_schema = '"+schema+"' " +\
            " AND    table_name   = '"+table+"' " +\
            " );"
    cur.execute(query)
    return cur.fetchall()[0][0]


def get_col_names(cur, table, schema):
    query = "SELECT column_name, data_type " + \
            " FROM information_schema.columns" + \
            " WHERE table_name = '" +table + "' AND table_schema = '" +schema + "';"

    cur.execute(query)
    rows = cur.fetchall()
    col_names = []
    for r in rows:
        col_names.append(r[0])
    return col_names


def db_types(t):
    type_mapping = {
        str: "TEXT",
        int: "INTEGER",
        float: "double precision",
        #time: "time",
        bool: "boolean"
    }

    return type_mapping[t]


def format_value(v):
    if v is None:
        v = "Null"
    elif isinstance(v, str):
        v = "'" + v + "'"
    elif isinstance(v, torch.Tensor):
        v = str(v.item())
    else:
        v = str(v)

    return v