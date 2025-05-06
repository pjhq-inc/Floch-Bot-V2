open System
open System.IO
open System.Text.RegularExpressions

let containsLink (line: string) =
    let pattern = @"https?://\S+"
    Regex.IsMatch(line, pattern)

let containsDate (line: string) =
    let pattern = @"^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APap][Mm]\] \w+"
    Regex.IsMatch(line, pattern)

let containsAttachmentOrEmbed (line: string) =
    line.Contains("{Attachment}") || line.Contains("{Embed}")

let removeDateAndName (line: string) =
    let pattern = @"^\[\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APap][Mm]\] \w+ "
    Regex.Replace(line, pattern, "")

let processFile (filePath: string) =
    let buffer = File.ReadAllText(filePath).Trim('\n', '\r')
    let lines = buffer.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
    
    let filteredLines =
        lines
        |> Array.filter (fun line -> not (containsLink line) && not (containsDate line) && not (containsAttachmentOrEmbed line) && not (String.IsNullOrWhiteSpace(line)))
        |> Array.map removeDateAndName

    let cleanedContent = String.Join("\n", filteredLines)
    File.WriteAllText(filePath, cleanedContent)

    printfn "File cleaned and saved."

let filePath = @"discord_clean.txt" //your filepath here
processFile filePath
