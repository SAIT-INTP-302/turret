{ pkgs, ... }:

{
  languages.python = {
    enable = true;
    venv = {
      enable = true;
      requirements = "-e .[dev]";
    };
    # Native libraries the opencv-python manylinux wheel dlopens
    libraries = [
      pkgs.libGL
      pkgs.glib
      pkgs.zlib
      pkgs.xorg.libxcb
      pkgs.xorg.libX11
      pkgs.xorg.libXext
      pkgs.xorg.libSM
      pkgs.xorg.libICE
    ];
  };
}
