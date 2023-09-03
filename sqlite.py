import sqlite3
import os

if __name__ == '__main__':
    if os.path.exists('reviews.sqlite'):
        os.remove('reviews.sqlite')
    conn = sqlite3.connect('reviews.sqlite')
    c = conn.cursor()
    c.execute('CREATE TABLE Recenzja_db'\
              ' (Recenzja TEXT, Sentyment INTEGER, Data TEXT)')
    example1 = 'I love this movie'
    c.execute("INSERT INTO Recenzja_db"\
              " (Recenzja, Sentyment, Data) VALUES"\
              " (?, ?, DATETIME('now'))", (example1, 1))
    example2 = 'I hate this movie'
    c.execute("INSERT INTO Recenzja_db" \
              " (Recenzja, Sentyment, Data) VALUES" \
              " (?, ?, DATETIME('now'))", (example2, 1))
    conn.commit()
    conn.close()