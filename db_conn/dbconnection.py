import mysql.connector as conx


# Conexion a la base de datos
def connect(hostserver, user, passw, db):
    global conn
    try:
        conn = conx.connect(
            host = hostserver,
            user = user,
            passwd = passw,
            database = db
        )
        if conn.is_connected():
            print('Connection OK !!')
            query("select * from clientes")
    except conn.Error as err:
        print('Error al conectar : ' + str(err))
    finally:
        conn.close()


# Metodo para ejecutar consultas por parametro.
def query(q_sql):
    try:        
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute(q_sql)
            result = cursor.fetchall()
            print(result)
    except conn.Error as err:
        print('Error al conectar : ' + str(err))
    finally:
        conn.close()


if __name__ == '__main__':
    connect('localhost', 'root', 'msquillari3186', 'test1')
