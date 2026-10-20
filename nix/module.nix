# NixOS module: run tyger-server (tyger.cpp turn-taking websocket server) as a
# systemd service. Options mirror the server's CLI:
#
#   services.tyger-server = {
#     enable = true;
#     model = "/var/lib/tyger/tyger-f16.gguf";
#     port = 8765;
#   };
#
# The model file must be readable by the service's dynamic user: any
# world-readable location (a nix store path is ideal), /var/lib/…, or similar —
# a protected per-user home dir will not work with the hardening applied here.
{ self }:
{
  config,
  lib,
  pkgs,
  utils,
  ...
}:
let
  inherit (lib)
    concatStringsSep
    mkEnableOption
    mkIf
    mkMerge
    mkOption
    optionals
    types
    ;

  cfg = config.services.tyger-server;
  lowPort = cfg.port < 1024;

  args =
    [
      "${cfg.package}/bin/tyger-server"
      "-m"
      (utils.escapeSystemdExecArg cfg.model)
      "--host"
      (utils.escapeSystemdExecArg cfg.host)
      "--port"
      (toString cfg.port)
    ]
    ++ optionals cfg.fast [ "--fast" ]
    ++ optionals cfg.cpuOnly [ "--cpu" ]
    ++ optionals cfg.noAsr [ "--no-asr" ]
    ++ optionals cfg.verbose [ "-v" ]
    ++ optionals (cfg.threads != null) [ "-t" (toString cfg.threads) ]
    ++ optionals (cfg.vulkanDevice != null) [ "--device" (toString cfg.vulkanDevice) ]
    ++ optionals (cfg.maxBatch != null) [ "--max-batch" (toString cfg.maxBatch) ]
    ++ cfg.extraArgs;
in
{
  options.services.tyger-server = {
    enable = mkEnableOption "the tyger.cpp realtime turn-taking websocket server";

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = "tyger.cpp package from this flake";
      description = "The tyger.cpp package providing the tyger-server binary.";
    };

    model = mkOption {
      type = types.path;
      description = ''
        Path to the tyger GGUF model file (tyger-f16.gguf is recommended for
        GPU, tyger-q8_0.gguf for CPU). Must be readable by the service's
        dynamic user — a path in the nix store is ideal.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Listen address. Use 0.0.0.0 to serve the LAN (combine with openFirewall).";
    };

    port = mkOption {
      type = types.port;
      default = 8765;
      description = "Listen port for HTTP and websockets.";
    };

    fast = mkOption {
      type = types.bool;
      default = false;
      description = "Default sessions to the cache-aware incremental encoder instead of the exact 8s window.";
    };

    cpuOnly = mkOption {
      type = types.bool;
      default = false;
      description = "Run on CPU only, skipping the Vulkan backend.";
    };

    noAsr = mkOption {
      type = types.bool;
      default = false;
      description = "Skip transcripts (turn events still fire).";
    };

    verbose = mkOption {
      type = types.bool;
      default = false;
      description = "Verbose logging to the journal.";
    };

    threads = mkOption {
      type = types.nullOr types.ints.positive;
      default = null;
      description = "CPU thread count (-t). null uses the server's default.";
      example = 8;
    };

    vulkanDevice = mkOption {
      type = types.nullOr types.ints.unsigned;
      default = null;
      description = "Vulkan device index (--device). null uses the server's default.";
    };

    maxBatch = mkOption {
      type = types.nullOr types.ints.positive;
      default = null;
      description = "Sessions batched per hop (--max-batch). null uses the server's default (8).";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open the port in the firewall.";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Extra raw arguments appended to the command line.";
    };

    extraEnvironment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      description = "Extra environment variables for the service.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.cpuOnly -> (cfg.vulkanDevice == null);
        message = "services.tyger-server.vulkanDevice makes no sense with cpuOnly = true.";
      }
    ];

    systemd.services.tyger-server = {
      description = "tyger.cpp realtime turn-taking websocket server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      restartTriggers = [ cfg.package cfg.model ];

      environment =
        mkMerge [
          cfg.extraEnvironment
          (mkIf (!cfg.cpuOnly) {
            # vulkan ICD manifests installed by hardware.graphics (incl. nvidia/mesa)
            XDG_DATA_DIRS = "/run/opengl-driver/share";
          })
        ];

      serviceConfig = {
        ExecStart = concatStringsSep " " args;
        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = 2;

        # GPU access: /dev/dri (mesa) and /dev/nvidia* are group-restricted
        SupplementaryGroups = mkIf (!cfg.cpuOnly) [ "render" "video" ];

        CapabilityBoundingSet = if lowPort then [ "CAP_NET_BIND_SERVICE" ] else [ "" ];
        AmbientCapabilities = mkIf lowPort [ "CAP_NET_BIND_SERVICE" ];

        RestrictAddressFamilies = [
          "AF_INET"
          "AF_INET6"
          "AF_UNIX"
        ];

        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = "read-only";
        RestrictSUIDSGID = true;
        LockPersonality = true;
        # no MemoryDenyWriteExecute / PrivateDevices: the vulkan drivers JIT
        # and need the GPU device nodes
        TimeoutStartSec = "5min"; # model files can be large
      };
    };

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [ cfg.port ];
  };
}