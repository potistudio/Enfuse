use iced::Subscription;
use iced::futures::SinkExt;
use midir::{Ignore, MidiInput};

/// FLX4 (DDJ-FLX4) のポート名に含まれるキーワード
const FLX4_KEYWORDS: &[&str] = &["FLX4", "FLX 4", "DDJ"];

/// MIDI 入力を購読する Subscription を返す。
///
/// FLX4 のポートを探して接続し、受信した生の MIDI メッセージを
/// `Message::Midi(Vec<u8>)` として流す。
pub fn listener<Message: 'static + Send>(
    on_midi: fn(Vec<u8>) -> Message,
) -> Subscription<Message> {
    Subscription::run_with_id(
        "flx4-midi-listener",
        iced::stream::channel(256, move |mut output| async move {
            // midir のコールバックは専用スレッドで同期的に呼ばれるので、
            // チャンネル経由で async 側へ橋渡しする。
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();

            // 接続は drop されると切れるので、ループが続く間スコープに保持する。
            let _conn = match connect(tx) {
                Some(conn) => conn,
                None => {
                    // FLX4 が見つからなければ何もせず待機（再接続は今は未対応）。
                    log::warn!("FLX4 MIDI port not found; listener is idle");
                    std::future::pending::<()>().await;
                    return;
                }
            };

            while let Some(bytes) = rx.recv().await {
                if output.send(on_midi(bytes)).await.is_err() {
                    break;
                }
            }
        }),
    )
}

/// FLX4 のポートを探して接続する。見つからなければ `None`。
fn connect(
    tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
) -> Option<midir::MidiInputConnection<()>> {
    let mut midi_in = MidiInput::new("Enfuse").ok()?;
    // ノブ等は大量に流れてくるので一切無視せず全部拾う。
    midi_in.ignore(Ignore::None);

    let ports = midi_in.ports();
    if ports.is_empty() {
        log::warn!("No MIDI input ports available");
        return None;
    }

    let mut target = None;
    for port in &ports {
        let name = midi_in.port_name(port).unwrap_or_default();
        log::info!("MIDI input port: {name}");
        if FLX4_KEYWORDS
            .iter()
            .any(|kw| name.to_uppercase().contains(&kw.to_uppercase()))
        {
            target = Some((port.clone(), name));
        }
    }

    let (port, name) = target?;
    log::info!("Connecting to FLX4 MIDI port: {name}");

    midi_in
        .connect(
            &port,
            "enfuse-flx4",
            move |_timestamp, message, _| {
                let _ = tx.send(message.to_vec());
            },
            (),
        )
        .ok()
}
