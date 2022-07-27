#!/bin/bash

# follow the instructions from https://github.com/cyberbotics/webots/wiki/Qt-compilation#linux to download and compile Qt before executing this script.

QT_VERSION=6.3.0
ICU_VERSION=68
QT_INSTALLATION_PATH=/opt/Software/haswell/Stages/2021a/software/Qt6/6.3.0-GCCcore-11.1.0
WEBOTS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}" )"/../.. && pwd)"

echo Installing Qt in Webots
echo Source: $QT_INSTALLATION_PATH
echo Destination: $WEBOTS_HOME

# uninstall any previous installation of Qt from Webots

cd $WEBOTS_HOME
rm -f bin/qt/lupdate
rm -f bin/qt/lrelease
rm -f bin/qt/moc
rm -rf include/qt
rm -rf lib/webots/qt
rm -rf lib/webots/libQt6*
rm -rf lib/webots/libicu*

# install Qt in Webots

mkdir include/qt
mkdir include/qt/QtConcurrent
mkdir include/qt/QtCore
mkdir include/qt/QtGui
mkdir include/qt/QtNetwork
mkdir include/qt/QtOpenGL
mkdir include/qt/QtOpenGLWidgets
mkdir include/qt/QtPrintSupport
mkdir include/qt/QtQml
mkdir include/qt/QtWebSockets
mkdir include/qt/QtWidgets
mkdir include/qt/QtXml
mkdir lib/webots/qt
mkdir lib/webots/qt/libexec
mkdir lib/webots/qt/plugins
mkdir lib/webots/qt/plugins/imageformats
mkdir lib/webots/qt/plugins/platforms
mkdir lib/webots/qt/plugins/platformthemes
mkdir lib/webots/qt/plugins/platforminputcontexts
mkdir lib/webots/qt/plugins/printsupport
mkdir lib/webots/qt/plugins/tls
mkdir lib/webots/qt/plugins/xcbglintegrations
mkdir lib/webots/qt/resources
mkdir lib/webots/qt/translations

#cp $QT_INSTALLATION_PATH/bin/lrelease                              bin/qt/
#cp $QT_INSTALLATION_PATH/bin/lupdate                               bin/qt/
cp $QT_INSTALLATION_PATH/lib/qt6/moc                               bin/qt/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtConcurrent                   include/qt/QtConcurrent/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtCore                         include/qt/QtCore/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtGui                          include/qt/QtGui/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtNetwork                      include/qt/QtNetwork/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtOpenGL                       include/qt/QtOpenGL/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtOpenGLWidgets                include/qt/QtOpenGLWidgets/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtPrintSupport                 include/qt/QtPrintSupport/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtQml                          include/qt/QtQml/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtWebSockets                   include/qt/QtWebSockets/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtWidgets                      include/qt/QtWidgets/
cp -r $QT_INSTALLATION_PATH/include/qt6/QtXml                          include/qt/QtXml/
rm -rf include/qt/Qt*/*/$QT_VERSION
cp $QT_INSTALLATION_PATH/lib/libQt6Concurrent.so.$QT_VERSION        lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6Core.so.$QT_VERSION              lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6DBus.so.$QT_VERSION              lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6Gui.so.$QT_VERSION               lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6Network.so.$QT_VERSION           lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6OpenGL.so.$QT_VERSION            lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6OpenGLWidgets.so.$QT_VERSION     lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6PrintSupport.so.$QT_VERSION      lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6Qml.so.$QT_VERSION               lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6WebSockets.so.$QT_VERSION        lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6Widgets.so.$QT_VERSION           lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6XcbQpa.so.$QT_VERSION            lib/webots/
cp $QT_INSTALLATION_PATH/lib/libQt6Xml.so.$QT_VERSION               lib/webots/
cp $QT_INSTALLATION_PATH/../../ICU/68.2-GCCcore-11.1.0/lib/libicudata.so.$ICU_VERSION.2           lib/webots/
cp $QT_INSTALLATION_PATH/../../ICU/68.2-GCCcore-11.1.0/lib/libicui18n.so.$ICU_VERSION.2           lib/webots/
cp $QT_INSTALLATION_PATH/../../ICU/68.2-GCCcore-11.1.0/lib/libicuuc.so.$ICU_VERSION.2             lib/webots/
echo $'[Paths]\nPrefix = ..\n' >                                    lib/webots/qt/libexec/qt.conf
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/platforms/libqxcb.so               lib/webots/qt/plugins/platforms/
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/platformthemes/libqgtk3.so         lib/webots/qt/plugins/platformthemes/
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/platforminputcontexts/libcomposeplatforminputcontextplugin.so lib/webots/qt/plugins/platforminputcontexts/
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/platforminputcontexts/libibusplatforminputcontextplugin.so    lib/webots/qt/plugins/platforminputcontexts/
#cp $QT_INSTALLATION_PATH/lib/qt6/plugins/printsupport/libcupsprintersupport.so lib/webots/qt/plugins/printsupport/
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/tls/*.so                           lib/webots/qt/plugins/tls/
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/xcbglintegrations/libqxcb-glx-integration.so lib/webots/qt/plugins/xcbglintegrations/
cp $QT_INSTALLATION_PATH/lib/qt6/plugins/imageformats/libqjpeg.so           lib/webots/qt/plugins/imageformats/
#cp -r $QT_INSTALLATION_PATH/translations/qt_*                       lib/webots/qt/translations/
#cp -r $QT_INSTALLATION_PATH/translations/qtbase_*                   lib/webots/qt/translations/
#cp -r $QT_INSTALLATION_PATH/translations/qtdeclarative_*            lib/webots/qt/translations/
#cp -r $QT_INSTALLATION_PATH/translations/qtwebsockets_*             lib/webots/qt/translations/

cd lib/webots
ln -sf libQt6Concurrent.so.$QT_VERSION        libQt6Concurrent.so.6
ln -sf libQt6Concurrent.so.$QT_VERSION        libQt6Concurrent.so
ln -sf libQt6Core.so.$QT_VERSION              libQt6Core.so.6
ln -sf libQt6Core.so.$QT_VERSION              libQt6Core.so
ln -sf libQt6DBus.so.$QT_VERSION              libQt6DBus.so.6
ln -sf libQt6DBus.so.$QT_VERSION              libQt6DBus.so
ln -sf libQt6Gui.so.$QT_VERSION               libQt6Gui.so.6
ln -sf libQt6Gui.so.$QT_VERSION               libQt6Gui.so
ln -sf libQt6Network.so.$QT_VERSION           libQt6Network.so.6
ln -sf libQt6Network.so.$QT_VERSION           libQt6Network.so
ln -sf libQt6OpenGL.so.$QT_VERSION            libQt6OpenGL.so.6
ln -sf libQt6OpenGL.so.$QT_VERSION            libQt6OpenGL.so
ln -sf libQt6OpenGLWidgets.so.$QT_VERSION     libQt6OpenGLWidgets.so.6
ln -sf libQt6OpenGLWidgets.so.$QT_VERSION     libQt6OpenGLWidgets.so
ln -sf libQt6PrintSupport.so.$QT_VERSION      libQt6PrintSupport.so.6
ln -sf libQt6PrintSupport.so.$QT_VERSION      libQt6PrintSupport.so
ln -sf libQt6Qml.so.$QT_VERSION               libQt6Qml.so.6
ln -sf libQt6Qml.so.$QT_VERSION               libQt6Qml.so
ln -sf libQt6WebSockets.so.$QT_VERSION        libQt6WebSockets.so.6
ln -sf libQt6WebSockets.so.$QT_VERSION        libQt6WebSockets.so
ln -sf libQt6Widgets.so.$QT_VERSION           libQt6Widgets.so.6
ln -sf libQt6Widgets.so.$QT_VERSION           libQt6Widgets.so
ln -sf libQt6Xml.so.$QT_VERSION               libQt6Xml.so.6
ln -sf libQt6Xml.so.$QT_VERSION               libQt6Xml.so
ln -sf libQt6XcbQpa.so.$QT_VERSION            libQt6XcbQpa.so.6
ln -sf libQt6XcbQpa.so.$QT_VERSION            libQt6XcbQpa.so
ln -sf libicudata.so.$ICU_VERSION.2           libicudata.so.$ICU_VERSION
ln -sf libicui18n.so.$ICU_VERSION.2           libicui18n.so.$ICU_VERSION
ln -sf libicuuc.so.$ICU_VERSION.2             libicuuc.so.$ICU_VERSION

cd ../..

#ARCHIVE=webots-qt-$QT_VERSION-linux64-release.tar.bz2
#echo Compressing $ARCHIVE \(please wait\)
#tar cjf $ARCHIVE lib/webots/libQt6* lib/webots/libicu* lib/webots/qt include/qt bin/qt/moc

echo Done.
