export interface ImageBehaviour {
    uri: string;
}

export type Align = 'left' | 'right' | 'center';

export interface TextBehaviour {
    text: string;
    align?: Align;
    color?: string;
    size?: number;
}

export interface EmbedBehaviour {
    url: string;
}

export interface AudioBehaviour {
    uri: string;
    name: string;
    loop?: boolean;
}

export interface BehaviourType {
    image?: ImageBehaviour;
    audio?: AudioBehaviour;
    text?: TextBehaviour;
    embed?: EmbedBehaviour;
    label: string;
}
