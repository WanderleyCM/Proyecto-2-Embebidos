# Recipe created by recipetool
# This is the basis of a recipe and may need further editing in order to be fully functional .
# ( Feel free to remove these comments when editing .)
# Unable to find any files that looked like license statements . Check the accompanying
# d oc um en ta ti on and source headers and set LICENSE and L I C _ F I L E S _ C H K S U M accordingly .
#
# NOTE : LICENSE is being set to " CLOSED " to allow you to at least start building - if
# this is not accurate with respect to the licensing of the software being built ( it
# will not be in most cases ) you must specify the correct value before using this
# recipe for anything other than initial testing / development !

LICENSE = "CLOSED"
LIC_FILES_CHKSUM = ""

# No information for SRC_URI yet ( only an external source tree was specified )

SRC_URI = "file://Aplicacion.py \
	file://modelo_caras.tflite"

# NOTE: no Makefile found , unable to determine what needs to be done

S = "${WORKDIR}"
TARGET_CC_ARCH += "${LDFLAGS}"

do_install () {
	install -d ${D}${bindir}
	install -m 0755 Aplicacion.py ${D}${bindir}
	install -m 0755 modelo_caras.tflite ${D}${bindir}
}

