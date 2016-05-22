import sys
import os
from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSlot
import json
import shutil


class foo(QWidget):

    def readConf(s):
        # back up the old file first
        shutil.copyfile('./conf.json', './conf.json.bk')
        s.conf = json.load(open('./conf.json'))

    def __init__(s):
        super(foo, s).__init__()
        try:
            s.readConf()
        except IOError:
            QMessageBox.warning(
                s, "NO CONFIG FILE!", "No such file or directory: './conf.json'", QMessageBox.Yes)
        s.initCompo()
        s.show()

    def initCompo(s):
        @pyqtSlot()
        def update_path():
            cur_path = os.path.abspath('..')
            path_txt.setText(cur_path)

        @pyqtSlot()
        def sendout_conf():
            s.conf["path"] = str(path_txt.text())
            s.conf["dpath"] = str(dpath_txt.text())
            s.conf["confpaths"] = [i.strip()
                                   for i in str(confs_txt.toPlainText()).split(';')]
            s.conf["confpaths"] = filter(
                lambda o: o != '', s.conf["confpaths"])

            for fpath in s.conf["confpaths"]:
                full_path = os.path.join(s.conf["path"], fpath, 'conf.json')
                json.dump(s.conf, open(full_path, 'w'))
                print 'ok conf.json sent to {0}'.format(full_path)
                sys.stdout.flush()
            print '\nALL OUT'
            sys.stdout.flush()

        def fillConf():
            path_txt.setText(s.conf["path"])
            dpath_txt.setText(s.conf["dpath"])
            confs_txt.setText(';\n'.join(s.conf["confpaths"]))

        # Buttons
        upd = QPushButton('Set root here')
        upd.setToolTip('Update root direcroty to be here')
        upd.clicked.connect(update_path)
        sout = QPushButton('Send-out')
        sout.setToolTip('Send out the modification.')
        sout.clicked.connect(sendout_conf)
        exit_btn = QPushButton('Exit')
        exit_btn.clicked.connect(exit)

        # Labels and Editors
        label1 = QLabel('Project Path')
        label2 = QLabel('Dataset Path')
        label3 = QLabel('Conf-file Paths')
        path_txt = QLineEdit()
        dpath_txt = QLineEdit()
        confs_txt = QTextEdit()

        # Grid Layout Arrangement
        main_grid = QGridLayout()
        main_grid.setSpacing(10)
        main_grid.addWidget(label1, 1, 0)
        main_grid.addWidget(label2, 2, 0)
        main_grid.addWidget(label3, 3, 0)
        main_grid.addWidget(path_txt, 1, 1)
        main_grid.addWidget(dpath_txt, 2, 1)
        main_grid.addWidget(confs_txt, 3, 1, 5, 1)
        main_grid.addWidget(upd, 1, 2)
        main_grid.addWidget(sout, 3, 2)
        main_grid.addWidget(exit_btn, 7, 2)
        # strech the column of grids with editors
        main_grid.setColumnStretch(1, 1)

        fillConf()
        s.setLayout(main_grid)
        s.setWindowTitle('Synchronize Conf-files')


def main():
    app = QApplication(sys.argv)
    w = foo()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
