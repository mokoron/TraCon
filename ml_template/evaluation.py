import numpy as np

from postgre_db import table_exists, db_types, get_col_names, format_value


class Evaluation:

    def __init__(self, config):
        self.result_types = {
            "speed": "speed"
        }
        self.table_name = {
            self.result_types["speed"]: config.speedResultTable
        }
        self.metrics = {
            self.result_types["speed"]: {}
        }
        self.data_types = {
            self.result_types["speed"]: {},
        }
        self.config = config

    def add_metric(self, name, type, result_type="speed"):
        self.data_types[result_type][name] = type

    def compute_metric(self, metrics, bestid=0, time_spent=0, weights=""):
        # I use this function for speed
        self.metrics[self.result_types["speed"]]["mae"] = metrics[0]
        self.metrics[self.result_types["speed"]]["mape"] = metrics[1]
        self.metrics[self.result_types["speed"]]["rmse"] = metrics[2]
        self.metrics[self.result_types["speed"]]["bestid"] = bestid
        self.metrics[self.result_types["speed"]]["time_spent"] = time_spent
        self.metrics[self.result_types["speed"]]["weights"] = weights

    def create_tables(self):
        with self.config.db.get_connection() as conn:
            with conn.cursor() as cur:
                # put a loop for each result_type
                for result_type in list(self.result_types.values()):
                    if not table_exists(cur, self.table_name[result_type], self.config.dbSchema):
                        create_query = "CREATE TABLE " + self.table_name[result_type] + " (" + \
                                       "config integer primary key references " + self.config.configTable + " "

                        for m in self.data_types[result_type]:
                            create_query += ", " + m + " " + db_types(self.data_types[result_type][m])
                        create_query += ");"

                        cur.execute(create_query)
                    else:
                        col_names = get_col_names(cur, self.table_name[result_type], self.config.dbSchema)
                        missing_rows = [x for x in self.data_types[result_type] if x not in col_names]

                        for mr in missing_rows:
                            query = "ALTER TABLE " + self.table_name[
                                result_type] + " ADD COLUMN " + mr + " " + db_types(
                                self.data_types[result_type][mr]) + ";"
                            cur.execute(query)

                    try:
                        view_query = "CREATE OR REPLACE VIEW " + f"view_{result_type}" + " as " + \
                                     "SELECT * FROM " + self.config.configTable + " c " + \
                                     " JOIN " + self.table_name[result_type] + " r on (c.id = r.config)" + \
                                     ";"

                        cur.execute(view_query)
                    except Exception as e:
                        raise Exception

    def store_results(self, result_type="speed"):  # result_type = "speed", "label", "classification"
        self.create_tables()
        with self.config.db.get_connection() as conn:
            with conn.cursor() as cur:

                cols = list(self.data_types[result_type].keys())
                query = "INSERT INTO " + self.table_name[result_type] + " (config," + (",".join(cols)) + ") VALUES ("
                try:
                    values = [format_value(self.config.id)]
                    for c in cols:
                        v = format_value(self.metrics[result_type][c])
                        values.append(v)  # v is of type tensor
                except Exception as e:
                    raise Exception

                query += ",".join(values)
                query += ")"

                cur.execute(query)
