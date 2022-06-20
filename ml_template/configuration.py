import configargparse
import copy
from postgre_db import table_exists, db_types, get_col_names, format_value

class Configuration:

    def __init__(self):
        self._parser = configargparse.ArgumentParser()

        # options parsed by the default parser
        self._options = None

        # individual configurations for different runs
        self._configs= []

        # arguments with more than one value
        self._multivalue_args = []

        # data types
        self._data_types = {}

        self._parser.add_argument("-d", "--defaultConfig", is_config_file=True, help="Default Config file")
        self._parser.add_argument("-c", "--config", is_config_file=True, help="Config file")

        # setup storage of results
        self._parser.add_argument("-dbH", "--dbHost", is_config_file=False, help="URL of postgre db")
        self._parser.add_argument("-dbU", "--dbUser", is_config_file=False, help="User of postgre db")
        self._parser.add_argument("-dbP", "--dbPassword", is_config_file=False, help="Password of postgre db")
        self._parser.add_argument("-dbN", "--dbName", is_config_file=False, help="Name of postgre db")
        self._parser.add_argument("-dbS", "--dbSchema", is_config_file=False, help="Schema to use in postgre db")
        self._parser.add_argument("-ct", "--configTable", is_config_file=False, help="Table to store configurations of runs")
        self._parser.add_argument("-srt", "--speedResultTable", is_config_file=False, help="Table to store speed results of runs")

    def add_entry(self, short, long, help, type=str, nargs='*'):
        self._parser.add("-"+short, "--"+long, help=help, is_config_file=False, type=type, nargs=nargs)
        self._data_types[long.lower()] = type

    def add_model_entry(self, long, help, action='store_false'):
        self._parser.add("--"+long, help=help, is_config_file=False, action=action)
        self._data_types[long.lower()] = bool

    def parse(self):
        self._options = self._parser.parse_args()

        # find values with more than one entry
        dict_options = vars(self._options)
        for k in dict_options :
            if isinstance(dict_options[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._options)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_options[ma]:
                    current = copy.deepcopy(c)
                    setattr(current, ma, v)
                    new_configs.append(current)

            # store splitted values
            self._configs = new_configs

    def create_config_table(self, cur):
        config_exists = table_exists(cur, self._options.configTable, self._options.dbSchema)

        if not config_exists:
            query = "CREATE TABLE "+self._options.configTable+" (" +\
                "    id serial primary key"

            for col in self._data_types:
                query += ", "+col+" "+db_types(self._data_types[col])

            query += ", start timestamp WITHOUT TIME ZONE default now())"

            cur.execute(query)
        else:
            col_names = get_col_names(cur, self._options.configTable, self._options.dbSchema)
            missing_rows = [x for x in self._data_types if x not in col_names]

            for mr in missing_rows:
                query = "ALTER TABLE " + self._options.configTable +" ADD COLUMN " + mr +" " + db_types(self._data_types[mr]) + ";"
                cur.execute(query)

    def setup_db(self, db):
        # create config tables

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                self.create_config_table(cur)

        for c in self._configs:
            c.db = db

    def get_configs(self):
        return self._configs


def register_run(config):

    lower_to_capitalized = {}
    for v in vars(config):
        lower_to_capitalized[v.lower()] = v

    with config.db.get_connection() as conn:
        with conn.cursor() as cur:
            col_names = get_col_names(cur, config.configTable, config.dbSchema)

            # remove internal columns
            col_names.remove("id")
            col_names.remove("start")

            query = "INSERT INTO "+config.configTable+" (" + (",".join(col_names)) + ") VALUES "
            values = []
            for col in col_names:
                attr = (getattr(config, lower_to_capitalized[col]))

                attr = format_value(attr)

                values.append(attr)

            query += "("+(",".join(values))+")"
            query += " RETURNING id"

            cur.execute(query)
            id = cur.fetchall()[0][0]
    return id