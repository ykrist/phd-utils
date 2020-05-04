#/bin/bash
set -e
PKG_ROOT_DIR=$(readlink -f $0 | xargs dirname)
PKG_BIN_DIR=$PKG_ROOT_DIR/bin
PKG_ENV_FILE_NAME="phd-utils_path_env.sh"
ACTIVATE_DIR=${CONDA_PREFIX}/etc/conda/activate.d
ACTIVATE_FILE=${ACTIVATE_DIR}/${PKG_ENV_FILE_NAME}
DEACTIVATE_DIR=${CONDA_PREFIX}/etc/conda/deactivate.d
DEACTIVATE_FILE=${DEACTIVATE_DIR}/${PKG_ENV_FILE_NAME}

function install {
  while true ; do
    read -sp "Install into $CONDA_PREFIX? [y/n] " RESPONSE
    echo
    case $RESPONSE in
      y) break ;;
      n) exit 0 ;;
    esac
  done
  conda develop .
  mkdir -p $ACTIVATE_DIR $DEACTIVATE_DIR

  ACTIVATE_CMD="export PATH=\$PATH:${PKG_BIN_DIR}:"
  DEACTIVATE_CMD="export PATH=\$(echo \$PATH | sed 's!:${PKG_BIN_DIR}:!!')"

  echo -e "#!/bin/sh\n${ACTIVATE_CMD}" >  $ACTIVATE_FILE
  echo -e "#!/bin/sh\n${DEACTIVATE_CMD}" > $DEACTIVATE_FILE
  echo "successfully installed."
  exit 0
};

function uninstall {
  conda develop -u $PKG_ROOT_DIR
  rm -f $ACTIVATE_FILE
  rm -f $DEACTIVATE_FILE
  echo "successfully uninstalled."
  exit 0
}

if [[ -z $CONDA_PREFIX ]] ; then
  echo "Error: not in a conda environment." >/dev/stderr
  exit 1
fi

case $1 in
  "install") install
    ;;
  "uninstall") uninstall
    ;;
  *) echo "Usage: ${0} (install | uninstall)"  >/dev/stderr ; exit 1
esac
