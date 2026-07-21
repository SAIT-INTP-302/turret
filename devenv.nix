{ pkgs, ... }:

{
  # Native libraries needed by the opencv-python manylinux wheel on Nix
  packages = [
    pkgs.libGL
    pkgs.glib
    pkgs.zlib
  ];

  languages.python = {
    enable = true;
    version = "3.12";
    venv = {
      enable = true;
      requirements = "-e .[dev]";
    };
  };
}
