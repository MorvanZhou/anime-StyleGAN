lastDir=`ls visual | sort -V | tail -n 1`
sz -y "visual/$lastDir/"train.log
sz -be "visual/$lastDir/"*
