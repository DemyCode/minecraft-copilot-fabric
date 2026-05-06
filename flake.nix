{
  description = "Minecraft Copilot Fabric Mod Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };

      nativeLibs = with pkgs; [
        # Graphics
        mesa
        libglvnd
        glfw
        xorg.libX11
        xorg.libXcursor
        xorg.libXrandr
        xorg.libXxf86vm
        # Audio
        libpulseaudio
        alsa-lib
        # Vulkan
        vulkan-loader
        # OpenMP — required by ONNX Runtime Java (libgomp.so)
        stdenv.cc.cc.lib
        # Narrator (flite TTS)
        flite
        # udev (hardware detection)
        udev
      ];
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "minecraft-copilot-dev";

        buildInputs = with pkgs; [
          jdk25
          gradle
          git
        ] ++ nativeLibs;

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath nativeLibs}:$LD_LIBRARY_PATH
          echo "Minecraft Copilot dev environment ready!"
          echo "  Run: ./gradlew runClient"
          echo "  Model files go in: run/config/minecraft-copilot/"
        '';
      };
    };
}
