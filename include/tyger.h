#ifndef TYGER_H
#define TYGER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef TYGER_BUILD
#    define TYGER_API __declspec(dllexport)
#  else
#    define TYGER_API __declspec(dllimport)
#  endif
#else
#  define TYGER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TYGER_API_VERSION 1

typedef struct tyger_model   tyger_model;
typedef struct tyger_session tyger_session;

enum tyger_backend {
    TYGER_BACKEND_AUTO   = 0,
    TYGER_BACKEND_CPU    = 1,
    TYGER_BACKEND_VULKAN = 2,
};

enum tyger_encoder_mode {
    TYGER_ENCODER_EXACT = 0, // re-encode the rolling 8 s window every hop, like the reference runtime
    TYGER_ENCODER_FAST  = 1, // cache-aware incremental encode, only the new frames each hop
};

enum tyger_mode {
    TYGER_MODE_LISTEN   = 0, // the agent is listening
    TYGER_MODE_SPEAKING = 1, // the agent is talking (barge-in detection)
};

enum tyger_event_kind {
    TYGER_EVENT_FRAME       = 0,
    TYGER_EVENT_ENDPOINT    = 1,
    TYGER_EVENT_BARGE       = 2,
    TYGER_EVENT_BACKCHANNEL = 3,
    TYGER_EVENT_PARTIAL     = 4,
};

typedef struct tyger_model_params {
    int  backend;     // enum tyger_backend
    int  device;      // gpu index for vulkan, -1 picks the first
    int  n_threads;   // cpu threads, 0 = auto
    bool enable_asr;  // load and run the rnnt decoder for transcripts
    bool verbose;
} tyger_model_params;

typedef struct tyger_session_params {
    float ep_thr;            // endpoint p_end threshold
    int   ep_deb;            // frames above it before ENDPOINT
    float barge_thr;         // yield threshold while speaking
    int   barge_deb;
    float bc_thr;            // backchannel threshold while listening
    int   bc_deb;
    int   bc_refractory;     // frames of quiet after a backchannel
    float theta;             // hazard threshold for the displayed state
    float partial_refresh_s; // how often PARTIAL transcripts refresh, <= 0 disables them
    int   fallback_sil_frames; // endpoint after this many silent frames even if p_end stays low, 0 = off
    int   encoder_mode;      // enum tyger_encoder_mode
    bool  emit_frames;       // include a FRAME event for every 80 ms frame
    int   prompt_id;         // asr language prompt, -1 = model default (auto)
} tyger_session_params;

typedef struct tyger_event {
    int         kind;       // enum tyger_event_kind
    double      t_s;        // session audio time at the end of this frame, seconds
    const char* state;      // FRAME: LISTEN, HOLD, SPEAK, BACKCHANNEL or idle
    float       p_end;
    float       yield_prob;
    float       hazard;
    float       bc_prob;
    float       onset_s;    // BARGE: how long the user had been talking
    float       turn_s;     // ENDPOINT, BACKCHANNEL: length of the turn so far
    int         segments;   // ENDPOINT: long-turn blocks that were transcribed early
    const char* text;       // ENDPOINT transcript, PARTIAL words so far, BACKCHANNEL partial
    const char* by;         // ENDPOINT: "head" or "silence"
} tyger_event;

TYGER_API const char * tyger_last_error(void);
TYGER_API const char * tyger_version(void);

TYGER_API tyger_model_params tyger_model_default_params(void);
TYGER_API tyger_model * tyger_model_load(const char * path, tyger_model_params params);
TYGER_API void          tyger_model_free(tyger_model * model);
TYGER_API const char  * tyger_model_info(const tyger_model * model); // json
TYGER_API int           tyger_sample_rate(const tyger_model * model);
TYGER_API int           tyger_hop_samples(const tyger_model * model);

TYGER_API tyger_session_params tyger_session_default_params(const tyger_model * model);
TYGER_API tyger_session * tyger_session_open(tyger_model * model, const tyger_session_params * params);
TYGER_API void            tyger_session_close(tyger_session * session);
TYGER_API void            tyger_session_reset(tyger_session * session); // a new turn, applied before the next hop
TYGER_API void            tyger_session_reset_keep(tyger_session * session); // same, but the last 2 s of user audio from speaking mode start the turn
TYGER_API int             tyger_session_set_params(tyger_session * session, const tyger_session_params * params);
TYGER_API int             tyger_session_get_params(const tyger_session * session, tyger_session_params * params);

// push any number of 16 kHz mono samples of both ears; every full 320 ms hop is processed.
// agent may be NULL (silence). returns the number of events, or -1 on error.
// events stay valid until the next feed or close on this session.
TYGER_API int tyger_session_feed(tyger_session * session, const float * user, const float * agent, int n_samples,
                                 int mode, const tyger_event ** events);

// one tick for many sessions at once (their encodes are batched). each gets exactly one hop.
// read each session's events with tyger_session_events afterwards.
TYGER_API int tyger_feed_batch(tyger_model * model, int n, tyger_session ** sessions, const float * const * user,
                               const float * const * agent, const int * modes);
TYGER_API int tyger_session_events(const tyger_session * session, const tyger_event ** events);

// the last feed's events as a json array, valid until the next feed
TYGER_API const char * tyger_session_events_json(tyger_session * session);

// offline transcription of a clip. returns a malloc'd utf-8 string (free with tyger_free) or NULL
TYGER_API char * tyger_transcribe(tyger_model * model, const float * pcm, int n_samples, int prompt_id);
TYGER_API void   tyger_free(void * p);

// debug: encode a clip offline to `n_layers` conformer layers with the given lookahead.
// writes up to max_frames * 1024 floats, returns the frame count or -1
TYGER_API int tyger_encode(tyger_model * model, const float * pcm, int n_samples, int n_layers, int lookahead,
                           float * out, int max_frames);

#ifdef __cplusplus
}
#endif

#endif
